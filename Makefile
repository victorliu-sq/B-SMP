.PHONY: all deps build run clean

deps:
	@bash ./deps/_scripts/install_all.sh

build:
	@bash ./expr/build_all.sh

run:
	@bash ./expr/run_bsmp.sh

all: deps build run

clean:
	@echo "[CLEAN] Clean Begins."
	@echo "[CLEAN] Remove installed dependencies under $(CURDIR)/deps ..."
	@rm -rf deps/bin deps/include deps/lib deps/lib64 deps/share deps/tmp deps/.stamp-*
	@echo "[CLEAN] Remove $(CURDIR)/bin ..."
	@rm -rf bin
	@echo "[CLEAN] Remove $(CURDIR)/build ..."
	@rm -rf build
	@echo "[CLEAN] Clean ."
