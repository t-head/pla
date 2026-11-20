// Private experiment. Bind externally allocated metadata to a native graph.
// Runtime contract: call shutdown_before_runtime_teardown() before Torch/PPU
// teardown. After shutdown, late callbacks intentionally retain their payloads
// until process exit; they must never enter a partially destroyed allocator.
#pragma once
#include <hggc_runtime_api.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace pla_graph_owner_experiment {

struct Stats {
  uint64_t created, attached, callbacks, released;
  uint64_t create_errors, attach_errors, release_errors, device_errors;
  bool shutdown;
};

class Owner {
  struct Node {
    std::shared_ptr<void> payload;
    int device;
    Node* next = nullptr;
    Owner* owner;
    Node(std::shared_ptr<void> value, int index, Owner* parent)
        : payload(std::move(value)), device(index), owner(parent) {}
  };

  std::atomic<Node*> retired_{nullptr};
  std::atomic<Node*> failed_device_{nullptr};
  std::atomic<bool> stopped_{false};
  std::atomic<uint64_t> created_{0}, attached_{0}, callbacks_{0}, released_{0};
  std::atomic<uint64_t> create_errors_{0}, attach_errors_{0}, release_errors_{0}, device_errors_{0};
  // Only host callers take this mutex. The runtime destructor never takes it.
  std::mutex attachment_mutex_;
  std::thread reaper_;

  static void enqueue(std::atomic<Node*>& queue, Node* node) noexcept {
    Node* previous = queue.load(std::memory_order_relaxed);
    do { node->next = previous; }
    while (!queue.compare_exchange_weak(previous, node,
             std::memory_order_release, std::memory_order_relaxed));
  }

  static void runtime_destructor(void* pointer) noexcept {
    auto* node = static_cast<Node*>(pointer);
    Owner* owner = node->owner;
    owner->callbacks_.fetch_add(1, std::memory_order_relaxed);
    // Atomic queue only: no allocation, mutex, Tensor/shared_ptr destruction,
    // runtime call, device selection, or waiting in the runtime callback.
    enqueue(owner->retired_, node);
  }

  void drain_once() noexcept {
    Node* node = retired_.exchange(nullptr, std::memory_order_acquire);
    while (node) {
      Node* next = node->next;
      // This is our independent host thread, outside the runtime callback.
      // Tensor/event deleters execute with their original logical device.
      if (hggcSetDevice(node->device) != hggcSuccess) {
        device_errors_.fetch_add(1, std::memory_order_relaxed);
        enqueue(failed_device_, node);  // Never destroy on a wrong device.
      } else {
        delete node;
        released_.fetch_add(1, std::memory_order_release);
      }
      node = next;
    }
  }

  void reap() noexcept {
    while (!stopped_.load(std::memory_order_acquire)) {
      drain_once();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // One finite final batch. Later callbacks remain in the process-lifetime
    // queue. Shutdown does not wait for or destroy still-live graphs.
    drain_once();
  }

  [[noreturn]] static void fail(const char* operation, hggcError_t status,
                              hggcError_t cleanup = hggcSuccess) {
    throw std::runtime_error(std::string(operation)+" failed: "+
        std::to_string(int(status))+", cleanup status "+std::to_string(int(cleanup)));
  }

 public:
  Owner() : reaper_([this] { reap(); }) {}
  Owner(const Owner&) = delete;
  Owner& operator=(const Owner&) = delete;
  // The singleton is deliberately never destroyed. A late runtime callback
  // must always find valid queue/counter storage, including interpreter exit.
  ~Owner() = delete;

  bool retain_if_capturing(hggcStream_t stream, int stream_device, int device,
                           std::shared_ptr<void> payload) {
    if (!payload || device < 0 || stream_device < 0 || stream_device != device) throw std::invalid_argument("invalid graph payload/device");
    std::lock_guard<std::mutex> guard(attachment_mutex_);
    if (stopped_.load(std::memory_order_acquire))
      throw std::runtime_error("graph payload owner has shut down");
    hggcStreamCaptureStatus status{};
    unsigned long long capture_id = 0;
    hggcGraph_t graph{};
    auto result = hggcStreamGetCaptureInfo(stream, &status, &capture_id, &graph,
                                          nullptr, nullptr, nullptr);
    if (result != hggcSuccess) fail("hggcStreamGetCaptureInfo", result);
    if (status == hggcStreamCaptureStatusNone) return false;
    if (status != hggcStreamCaptureStatusActive || !graph || !capture_id)
      throw std::runtime_error("graph capture is not active/valid");
    // The bridge obtains raw handle and stream_device from the same typed
    // c10::cuda::CUDAStream. SDK StreamGetDevice returned 998 in real capture;
    // never substitute an unverified guessed device for typed provenance.
    int current_device = -1;
    result = hggcGetDevice(&current_device);
    if (result != hggcSuccess) fail("hggcGetDevice", result);
    if (stream_device != device || current_device != device)
      throw std::runtime_error("graph stream, payload and current devices must match");
    std::unique_ptr<Node> node(new Node(std::move(payload), device, this));
    hggcUserObject_t object{};
    result = hggcUserObjectCreate(&object, node.get(), runtime_destructor, 1,
                                   hggcUserObjectNoDestructorSync);
    if (result != hggcSuccess) {
      create_errors_.fetch_add(1, std::memory_order_relaxed);
      fail("hggcUserObjectCreate", result);  // Host-owned unique_ptr rolls back.
    }
    created_.fetch_add(1, std::memory_order_relaxed);
    // The runtime owns the Node from Create success, including attach failure.
    // Releasing the user reference is the only permitted cleanup thereafter.
    node.release();
    result = hggcGraphRetainUserObject(graph, object, 1, hggcGraphUserObjectMove);
    if (result != hggcSuccess) {
      attach_errors_.fetch_add(1, std::memory_order_relaxed);
      const auto cleanup = hggcUserObjectRelease(object, 1);
      if (cleanup != hggcSuccess) release_errors_.fetch_add(1, std::memory_order_relaxed);
      // If Release fails, retain the runtime-owned allocation rather than
      // risking a callback accessing freed memory. This is an explicit error.
      fail("hggcGraphRetainUserObject", result, cleanup);
    }
    attached_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  Stats stats() const noexcept {
    return {created_.load(), attached_.load(), callbacks_.load(), released_.load(),
            create_errors_.load(), attach_errors_.load(), release_errors_.load(),
            device_errors_.load(), stopped_.load()};
  }

  bool wait_for_releases(uint64_t count, unsigned timeout_ms) const noexcept {
    const auto deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
    while (released_.load(std::memory_order_acquire) < count) {
      if (stopped_.load(std::memory_order_acquire) || device_errors_.load() ||
          std::chrono::steady_clock::now() >= deadline) return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
  }

  void shutdown_before_runtime_teardown() {
    std::lock_guard<std::mutex> guard(attachment_mutex_);
    if (!stopped_.exchange(true, std::memory_order_acq_rel)) reaper_.join();
  }
};

inline Owner& owner() {
  static Owner* value = new Owner();
  return *value;
}

}  // namespace pla_graph_owner_experiment
