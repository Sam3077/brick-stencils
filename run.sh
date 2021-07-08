
nvcc laplacian-stencils.cu ../bricklib/build/src/libbrickhelper.a -I ../bricklib/include -Xcompiler -fopenmp -O3 -o stencils && \
srun ncu --section=FlopsSection --section=SchedulerStats --section=LaunchStats --section=SpeedOfLight --section=MemoryWorkloadAnalysis --section-folder-recursive=sections --section=WarpStateStats --kernel-name=regex:'^(?!no_prof).*$' stencils