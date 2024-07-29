#include <chrono>
#include <ostream>

#include <ATen/ScalarOps.h>
#include <ATen/record_function.h>
#include <c10/core/TensorImpl.h>
// using torch_npu acl headers in stead of cann's
// pre include before hccl/hccl.h to prevent mismatch between two vesions of acl.h
#include <third_party/acl/inc/acl/acl.h>
#include <hccl/hccl.h>

#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/npu_log.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUEventManager.h>
#include <torch_npu/csrc/core/npu/NPUException.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/NPUTensorImpl.h>
#include <torch_npu/csrc/framework/OpParamMaker.h>

#include "torch_npu_utils.hpp"

std::unordered_map<SubModule, std::string> submoduleMap = {
    {SubModule::PTA, "PTA"},
    {SubModule::OPS, "OPS"},
    {SubModule::DIST, "DIST"},
    {SubModule::GRAPH, "GRAPH"},
    {SubModule::PROF, "PROF"}
};

std::unordered_map<ErrCode, std::string> errCodeMap = {
    {ErrCode::SUC, "success"},
    {ErrCode::PARAM, "invalid parameter"},
    {ErrCode::TYPE, "invalid type"},
    {ErrCode::VALUE, "invalid value"},
    {ErrCode::PTR, "invalid pointer"},
    {ErrCode::INTERNAL, "internal error"},
    {ErrCode::MEMORY, "memory error"},
    {ErrCode::NOT_SUPPORT, "feature not supported"},
    {ErrCode::NOT_FOUND, "resource not found"},
    {ErrCode::UNAVAIL, "resource unavailable"},
    {ErrCode::SYSCALL, "system call failed"},
    {ErrCode::TIMEOUT, "timeout error"},
    {ErrCode::PERMISSION, "permission error"},
    {ErrCode::ACL, "call acl api failed"},
    {ErrCode::HCCL, "call hccl api failed"},
    {ErrCode::GE, "call ge api failed"}
};

static std::string getCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());

    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm* timeInfo = std::localtime(&currentTime);

    auto milli_time = std::chrono::duration_cast<std::chrono::milliseconds>(micros).count() % 1000;
    auto micro_time = micros.count() % 1000;

    std::ostringstream oss;
    oss << std::put_time(timeInfo, "%Y-%m-%d-%H:%M:%S");
    return oss.str();
}

std::string formatErrorCode(SubModule submodule, ErrCode errorCode)
{
    std::ostringstream oss;
    int deviceIndex = -1;
    c10_npu::GetDevice(&deviceIndex);
    char* rankId_val = std::getenv("RANK");
    int64_t rank_id = (rankId_val != nullptr) ? strtol(rankId_val, nullptr, 10) : -1;
    oss << "\n[ERROR] " << getCurrentTimestamp() << " (PID:" << getpid() << ", Device:" << deviceIndex << ", RankID:" << rank_id << ") ";
    oss << "ERR" << std::setw(2) << std::setfill('0') << static_cast<int>(submodule);
    oss << std::setw(3) << std::setfill('0') << static_cast<int>(errorCode);
    oss << " " << submoduleMap[submodule] << " " << errCodeMap[errorCode];
    return oss.str();
}

namespace torch_npu {
    // NPUStorageImpl
    void NPUStorageImpl::release_resources() { StorageImpl::release_resources(); }
    NPUStorageImpl::NPUStorageImpl(use_byte_size_t use_byte_size, size_t size_bytes, at::DataPtr data_ptr, at::Allocator* allocator, bool resizable)
        : c10::StorageImpl(use_byte_size, size_bytes, at::DataPtr(std::move(data_ptr)), allocator, resizable) {}
    // NPUTensorImpl
    NPUTensorImpl::NPUTensorImpl(c10::Storage &&storage, const caffe2::TypeMeta &data_type)
        : c10::TensorImpl(std::move(storage),
                          c10::DispatchKeySet{c10::DispatchKey::PrivateUse1,
                                            c10::DispatchKey::AutogradPrivateUse1},
                          data_type)
    {
        is_non_overlapping_and_dense_ = false;
    }
    void NPUTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {};
    c10::intrusive_ptr<c10::TensorImpl> NPUTensorImpl::shallow_copy_and_detach(
        const c10::VariableVersion& version_counter,
        bool allow_tensor_metadata_change) const {};
    c10::intrusive_ptr<c10::TensorImpl> NPUTensorImpl::shallow_copy_and_detach(
        c10::VariableVersion&& version_counter,
        bool allow_tensor_metadata_change) const {};
    NPUTensorImpl::~NPUTensorImpl() {};
} // namespace torch_npu

/*
namespace c10_npu {
// NPUEventManager
NPUEventManager::NPUEventManager() : thread_pool_(std::make_shared<c10::TaskThreadPool>(5)){};

NPUEventManager &NPUEventManager::GetInstance()
{
    static NPUEventManager instance;
    return instance;
}

void NPUEventManager::run(aclrtEvent event)
{
    int err = aclrtDestroyEvent(event);
    if (err != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
        return;
    }
    ASCEND_LOGI("Event: aclrtDestroyEvent is successfully executed, event=%p", event);
}

aclError NPUEventManager::QueryAndDestroyEvent()
{
    std::lock_guard<std::mutex> guard(event_queue_mutex_);
    while (!npu_events_.empty()) {
        aclrtEvent event = npu_events_.front();
        acl::aclrtEventRecordedStatus recordStatus = acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
        aclError err = acl::AclQueryEventRecordedStatus(event, &recordStatus);
        if (err != ACL_ERROR_NONE) {
            C10_NPU_SHOW_ERR_MSG();
            return err;
        }
        if (recordStatus != acl::ACL_EVENT_RECORDED_STATUS_COMPLETE) {
            break;
        } else {
            acl::aclrtEventWaitStatus waitStatus = acl::ACL_EVENT_WAIT_STATUS_RESERVED;
            // if the event usage is unknown, ensure the event id not destroyed in advance.
            aclError err_wait = acl::AclQueryEventWaitStatus(event, &waitStatus);
            if (err_wait != ACL_ERROR_NONE) {
                C10_NPU_SHOW_ERR_MSG();
                return err_wait;
            }
            if (waitStatus != acl::ACL_EVENT_WAIT_STATUS_COMPLETE) {
                break;
            }
        }
        {
            thread_pool_->run(std::bind(&NPUEventManager::run, this, event));
        }

        npu_events_.pop_front();
    }
    return ACL_ERROR_NONE;
}

aclError NPUEventManager::LazyDestroy(aclrtEvent npu_event)
{
    if (c10_npu::acl::IsExistCreateEventExWithFlag()) {
        int err = aclrtDestroyEvent(npu_event);
        if (err == ACL_ERROR_NONE) {
            ASCEND_LOGI("Event: aclrtDestroyEvent is successfully executed, event=%p", npu_event);
        }
        return err;
    }
    std::lock_guard<std::mutex> guard(event_queue_mutex_);
    npu_events_.push_back(npu_event);
    return ACL_ERROR_NONE;
}

void NPUEventManager::ClearEvent()
{
    if (thread_pool_ != nullptr) {
        thread_pool_->waitWorkComplete();
    }

    while (!npu_events_.empty()) {
        aclrtEvent event = npu_events_.front();
        auto err = aclrtDestroyEvent(event);
        if (err != ACL_ERROR_NONE) {
            NPU_CHECK_WARN(err);
        } else {
            ASCEND_LOGI("Event: aclrtDestroyEvent is successfully executed, event=%p", event);
        }
        npu_events_.pop_front();
    }
}
void NPUEventManager::IncreaseUnrecordedCount(aclrtEvent event)
{
    std::lock_guard<std::mutex> guard(event_unrecorded_count_mutex_);

    auto it = event_unrecorded_count_.find(event);
    if (it != event_unrecorded_count_.end()) {
        it->second++;
        ASCEND_LOGI("Event: unrecorded count increase, now=%d.", it->second);
    } else {
        event_unrecorded_count_.insert(std::pair<aclrtEvent, int>(event, 1));
        ASCEND_LOGI("Event: unrecorded count increase, now=%d.", 1);
    }
}

void NPUEventManager::DecreaseUnrecordedCount(aclrtEvent event)
{
    std::lock_guard<std::mutex> guard(event_unrecorded_count_mutex_);

    auto it = event_unrecorded_count_.find(event);
    TORCH_CHECK(
        it != event_unrecorded_count_.end(),
        "Event: event must enqueue before dequeue, event=",
        (void *) event, PTA_ERROR(ErrCode::INTERNAL));
    if (it->second == 1) {
        event_unrecorded_count_.erase(event);
        ASCEND_LOGI("Event: unrecorded count decrease, now=%d.", 0);
    } else {
        it->second--;
        ASCEND_LOGI("Event: unrecorded count decrease, now=%d.", it->second);
    }
}

bool NPUEventManager::IsEventRecorded(aclrtEvent event)
{
    std::lock_guard<std::mutex> guard(event_unrecorded_count_mutex_);

    auto it = event_unrecorded_count_.find(event);
    return it == event_unrecorded_count_.end();
}

// AsyncTaskQueueInterface
// namespace queue {
// std::atomic<uint64_t> QueueParas::g_correlation_id{0};
// std::map<int64_t, std::string> CopyParas::COPY_PARAS_MAP{
//   {ACL_MEMCPY_HOST_TO_HOST, "acl_memcpy_host_to_host"},
//   {ACL_MEMCPY_HOST_TO_DEVICE, "acl_memcpy_host_to_device"},
//   {ACL_MEMCPY_DEVICE_TO_HOST, "acl_memcpy_device_to_host"},
//   {ACL_MEMCPY_DEVICE_TO_DEVICE, "acl_memcpy_device_to_device"},
// };
// std::map<int64_t, std::string> EventParas::EVENT_PARAS_MAP{
//     {RECORD_EVENT, "record_event"},
//     {WAIT_EVENT, "wait_event"},
//     {LAZY_DESTROY_EVENT, "destroy_event"},
// };
// void CopyParas::Copy(CopyParas& other) {
//   this->dst = other.dst;
//   this->dstLen = other.dstLen;
//   this->src = other.src;
//   this->srcLen = other.srcLen;
//   this->kind = other.kind;
// }

// void EventParas::Copy(EventParas& other) {
//   this->event = other.event;
//   this->eventAllocatorType = other.eventAllocatorType;
// }

// class AsyncCopyTask {
// public:
//   AsyncCopyTask(
//       void* dst,
//       size_t dstLen,
//       void* src,
//       size_t srcLen,
//       aclrtMemcpyKind kind);
//   ~AsyncCopyTask() = default;
//   void LaunchCopyTask();

// private:
//   CopyParas copyParam_;
// };

// class EventTask {
// public:
//   explicit EventTask(
//       aclrtEvent event,
//       EventAllocatorType allocatorType = RESERVED)
//       : eventParam_(event, allocatorType){};
//   ~EventTask() = default;
//   void LaunchRecordTask(
//       c10_npu::NPUStream npuStream);
//   void LaunchWaitTask(c10_npu::NPUStream npuStream);
//   void LaunchLazyDestroyTask(c10::DeviceIndex device_index);

// private:
//   EventParas eventParam_;
// };

// AsyncCopyTask::AsyncCopyTask(
//     void* dst,
//     size_t dstLen,
//     void* src,
//     size_t srcLen,
//     aclrtMemcpyKind kind) {
//   copyParam_.dst = dst;
//   copyParam_.dstLen = dstLen;
//   copyParam_.src = src;
//   copyParam_.srcLen = srcLen;
//   copyParam_.kind = kind;
// }

// void AsyncCopyTask::LaunchCopyTask() {
//   RECORD_FUNCTION(CopyParas::COPY_PARAS_MAP[copyParam_.kind], std::vector<c10::IValue>({}));
//   if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
//     QueueParas params(ASYNC_MEMCPY, sizeof(CopyParas), &copyParam_);
//     c10_npu::enCurrentNPUStream(&params);
//   } else {
//     c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
//     NPU_CHECK_ERROR(aclrtMemcpyAsync(
//         copyParam_.dst,
//         copyParam_.dstLen,
//         copyParam_.src,
//         copyParam_.srcLen,
//         copyParam_.kind,
//         stream));
//   }
// }

// aclError LaunchAsyncCopyTask(
//     void* dst,
//     size_t dstLen,
//     void* src,
//     size_t srcLen,
//     aclrtMemcpyKind kind) {
//   AsyncCopyTask copyTask(dst, dstLen, src, srcLen, kind);
//   copyTask.LaunchCopyTask();
//   return ACL_ERROR_NONE;
// }

// void EventTask::LaunchRecordTask(c10_npu::NPUStream npuStream) {
//     RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[RECORD_EVENT], std::vector<c10::IValue>({}));
//   if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
//     c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
//     c10_npu::setCurrentNPUStream(npuStream);
//     QueueParas params(RECORD_EVENT, sizeof(EventParas), &eventParam_);
//     c10_npu::NPUEventManager::GetInstance().IncreaseUnrecordedCount(eventParam_.event);
//     c10_npu::enCurrentNPUStream(&params);
//     c10_npu::setCurrentNPUStream(currentStream);
//     ASCEND_LOGI("Event: LaunchRecordTask is successfully executed, event=%p", eventParam_.event);
//   } else {
//     NPU_CHECK_ERROR(aclrtRecordEvent(eventParam_.event, npuStream));
//     ASCEND_LOGI("Event: aclrtRecordEvent is successfully executed, stream=%p, event=%p", npuStream.stream(false), eventParam_.event);
//   }
// }

// aclError LaunchRecordEventTask(aclrtEvent event, c10_npu::NPUStream npuStream) {
//   EventTask recordTask(event);
//   recordTask.LaunchRecordTask(npuStream);
//   return ACL_ERROR_NONE;
// }

// void EventTask::LaunchWaitTask(c10_npu::NPUStream npuStream) {
//     RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[WAIT_EVENT], std::vector<c10::IValue>({}));
//   if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
//     c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
//     c10_npu::setCurrentNPUStream(npuStream);
//     QueueParas params(WAIT_EVENT, sizeof(EventParas), &eventParam_);
//     c10_npu::enCurrentNPUStream(&params);
//     c10_npu::setCurrentNPUStream(currentStream);
//     ASCEND_LOGI("Event: LaunchWaitTask is successfully executed, event=%p", eventParam_.event);
//   } else {
//     NPU_CHECK_ERROR(aclrtStreamWaitEvent(npuStream, eventParam_.event));
//     ASCEND_LOGI("Event: aclrtStreamWaitEvent is successfully executed, stream=%p, event=%p", npuStream.stream(false), eventParam_.event);
//   }
// }

// aclError LaunchWaitEventTask(aclrtEvent event, c10_npu::NPUStream npuStream) {
//   EventTask waitTask(event);
//   waitTask.LaunchWaitTask(npuStream);
//   return ACL_ERROR_NONE;
// }

// void EventTask::LaunchLazyDestroyTask(c10::DeviceIndex device_index) {
//     RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[LAZY_DESTROY_EVENT], std::vector<c10::IValue>({}));
//   if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
//     QueueParas params(LAZY_DESTROY_EVENT, sizeof(EventParas), &eventParam_);
//     c10_npu::enCurrentNPUStream(&params, device_index);
//     ASCEND_LOGI("Event: LaunchLazyDestroyTask is successfully executed, event=%p", eventParam_.event);
//   } else {
//     NPU_CHECK_ERROR(c10_npu::NPUEventManager::GetInstance().LazyDestroy(
//         eventParam_.event));
//   }
// }

// aclError LaunchLazyDestroyEventTask(aclrtEvent event, c10::DeviceIndex device_index) {
//   EventTask lazyDestroyTask(event);
//   lazyDestroyTask.LaunchLazyDestroyTask(device_index);
//   return ACL_ERROR_NONE;
// }
// } // namespace queue

// AclInterface
#ifdef GET_FUNC
#define ORIGIN_GET_FUNC GET_FUNC
#undef GET_FUNC
#endif
#define GET_FUNC(x) x

namespace acl {

bool IsExistCreateEventExWithFlag()
{
    typedef aclError(*AclrtCreateEventWithFlagFunc)(aclrtEvent*, uint32_t);
    static AclrtCreateEventWithFlagFunc func = (AclrtCreateEventWithFlagFunc)GET_FUNC(aclrtCreateEventExWithFlag);
    return func != nullptr;
}

aclError AclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag)
{
    typedef aclError(*AclrtCreateEventWithFlagFunc)(aclrtEvent*, uint32_t);
    // Recommend aclrtCreateEventExWithFlag.
    // Differences from aclrtCreateEventWithFlag:
    //   1. Event can be reused naturally, aclrtResetEvent is not supported.
    //   2. There is no limit on the number of events.
    //   3. Only support query event record status, aclrtQueryEvent and aclrtQueryEventWaitStatus are not supported.
    //   4. aclrtDestroyEvent change to asynchronous destroy event.
    static AclrtCreateEventWithFlagFunc func = (AclrtCreateEventWithFlagFunc)GET_FUNC(aclrtCreateEventExWithFlag);
    if (func == nullptr) {
        TORCH_NPU_WARN_ONCE(func, "Failed to find function ", "aclrtCreateEventExWithFlag");
        func = (AclrtCreateEventWithFlagFunc)GET_FUNC(aclrtCreateEventWithFlag);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclrtCreateEventWithFlag", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(event, flag);
}
} // namespace acl

#undef GET_FUNC
#ifdef ORIGIN_GET_FUNC
#define GET_FUNC ORIGIN_GET_FUNC
#undef ORIGIN_GET_FUNC
#endif

// NPUStream
NPUStream getStreamFromPool(const bool isHighPriority, c10::DeviceIndex device_index) {
    // fake implementation: using getNPUStreamFromPool
    return getNPUStreamFromPool(device_index);
}

}  // namespace c10_npu
*/

namespace infer_ext {
namespace ascend {

aclDataType convert_to_acl_data_type(const at::ScalarType &data_type) {
    auto acl_dtype =
        kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED,
        std::string(c10::toString(data_type)) + " has not been supported", OPS_ERROR(ErrCode::NOT_SUPPORT))
    return acl_dtype;
}

bool is_scalar_wrapped_to_tensor(const at::Tensor &tensor) {
    if (tensor.dim() == 0 && !torch_npu::utils::is_npu(tensor)) {
        return true;
    }
    return false;
}

at::Tensor copy_scalar_to_device(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type) {
    at::Tensor cpu_tensor = scalar_to_tensor(cpu_scalar).to(scalar_data_type);
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&deviceIndex));
    return cpuPinMemTensor.to(
        c10::Device(c10::DeviceType::PrivateUse1, deviceIndex),
        cpuPinMemTensor.scalar_type(), true, true);
}

at::Tensor unsafe_empty_workspace(uint64_t workspace_size) {
    ASCEND_LOGD("Alloc workspace %zu bytes unsafely.", workspace_size);
    c10::Allocator *allocator = c10_npu::NPUCachingAllocator::get();
    c10::intrusive_ptr<c10::StorageImpl> storage_impl =
        c10::make_intrusive<torch_npu::NPUStorageImpl>(
        c10::StorageImpl::use_byte_size_t(), workspace_size,
        allocator->allocate(workspace_size), allocator, true);
    static auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(at::kByte));
    auto tensor = at::detail::make_tensor<torch_npu::NPUTensorImpl>(
        storage_impl, dtype);
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(c10::MemoryFormat::Contiguous);
    return tensor;
}

#define GET_FUNC(funcName) funcName

aclError AclSetCompileopt(aclCompileOpt opt, const char* value) {
    typedef aclError (*aclSetCompileoptFunc)(aclCompileOpt opt, const char* value);
    static aclSetCompileoptFunc func = nullptr;
    if (func == nullptr) {
        func = (aclSetCompileoptFunc)GET_FUNC(aclSetCompileopt);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclSetCompileopt");
    auto ret = func(opt, value);
    return ret;
}

aclError AclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value) {
    typedef aclError (*AclrtCtxSetSysParamOptFunc)(aclSysParamOpt opt, int64_t value);
    static AclrtCtxSetSysParamOptFunc func = nullptr;
    if (func == nullptr) {
        // func = (AclrtCtxSetSysParamOptFunc)GET_FUNC(aclrtCtxSetSysParamOpt);
    }
    if (func == nullptr) {
        TORCH_WARN("Failed to find this aclrtCtxSetSysParamOpt function!");
        return ACL_ERROR_NONE;
    }
    auto ret = func(opt, value);
    return ret;
}

#define HCCL_CHECK_ERROR(cmd)                                         \
    do {                                                              \
        HcclResult error = cmd;                                       \
        if (error != HCCL_SUCCESS) {                                  \
            std::string err = "[ERROR] HCCL error in: " +             \
                std::string(__FILE__) +                               \
                 ":" + std::to_string(__LINE__) +                     \
                DIST_ERROR(ErrCode::HCCL) + ".\n";                    \
            throw std::runtime_error(err);                            \
        }                                                             \
    } while (0)

void SetDeterministic() {
    auto deterministicAlgorithmsStatus = at::globalContext().deterministicAlgorithms();
    if (at_npu::native::deterministicaclnn_oldstatus != deterministicAlgorithmsStatus) {
        NPU_CHECK_ERROR(
            AclSetCompileopt(aclCompileOpt::ACL_OP_DETERMINISTIC, deterministicAlgorithmsStatus ? "1" : "0"));
        NPU_CHECK_ERROR(
            AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, deterministicAlgorithmsStatus ? 1 : 0));
        HcclConfigValue configValue = {deterministicAlgorithmsStatus ? 1 : 0};
        HCCL_CHECK_ERROR(HcclSetConfig(HcclConfig::HCCL_DETERMINISTIC, configValue));
        at_npu::native::deterministicaclnn_oldstatus = deterministicAlgorithmsStatus;
    }
}

} // namespace ascend
} // namespace infer_ext
