red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@

# Extraction and slicing for Segthor
data/segthor_train: data/segthor_train.zip
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/segthor_train.sha256
	unzip -q $<
	rm -f $@/.DS_STORE

data/SEGTHOR: data/segthor_train
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir $^ --dest_dir $@_tmp \
		--shape 256 256 --retain 10
	mv $@_tmp $@

# Extraction and slicing for Segthor test data
data/segthor_test: data/test.zip
	$(info $(yellow)unzip $< to $@$(reset))
	sha256sum -c data/test.zip.sha256
	unzip $< -d $@
	rm -f $@/.DS_STORE

data/SEGTHOR_TEST: data/segthor_test
	$(info $(green)python $(CFLAGS) slice_segthor_test.py$(reset))
	python $(CFLAGS) slice_segthor_test.py --source_dir $< --dest_dir $@_tmp \
		--shape 256 256

