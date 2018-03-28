generated_files := \
         nvvl/src/nvvl_generated.h

default: $(generated_files)

nvvl/src/nvvl_generated.h: ../include/VideoLoader.h ../include/PictureSequence.h
	echo '#include "$<"' | gcc -E -P -DCFFI -o $@ -

.PHONY: clean
clean:
	rm $(generated_files)
