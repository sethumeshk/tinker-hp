HIPCC        ?= hipcc
CXXFLAGS     ?= -O2 -std=c++17
ARCH         ?= gfx90a
TARGET       ?= hip_rocprofv3_synthetic
SRC          := hip_rocprofv3_synthetic.cpp
LDFLAGS      ?= -lrocblas -lroctx64

.PHONY: all clean run profile-trace profile-stats help

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) --offload-arch=$(ARCH) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) 1048576 5

profile-trace: $(TARGET)
	rocprofv3 --hip-trace --hsa-trace --rocblas-trace --kernel-trace --marker-trace --output-format pftrace --output-file rocprof_trace ./$(TARGET) 1048576 5

profile-stats: $(TARGET)
	rocprofv3 --stats --hip-trace --kernel-trace --rocblas-trace --output-file rocprof_stats ./$(TARGET) 1048576 5

clean:
	rm -f $(TARGET) rocprof_trace* rocprof_stats*

help:
	@echo "Targets:"
	@echo "  make                # build for MI250X (gfx90a)"
	@echo "  make run            # execute synthetic workload"
	@echo "  make profile-trace  # generate Perfetto-compatible trace"
	@echo "  make profile-stats  # generate summary statistics"
	@echo "  make clean"
	@echo ""
	@echo "After profile-trace, open with https://ui.perfetto.dev and load rocprof_trace.pftrace"
