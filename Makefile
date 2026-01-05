ROOT_PATH = .

all: ds-build cours-build

clean-all: ds-clean td-clean cours-clean

ds-build:
	$(MAKE) -C $(ROOT_PATH)/ds all

ds-clean:
	$(MAKE) -C $(ROOT_PATH)/ds clean-all

cours-build:
	$(MAKE) -C $(ROOT_PATH)/cours all

cours-clean:
	$(MAKE) -C $(ROOT_PATH)/cours clean-all


.PHONY: all clean-all ds-build cours-build ds-clean cours-clean
