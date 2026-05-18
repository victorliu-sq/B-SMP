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
	@rm -rf deps/bin deps/include deps/lib deps/lib64 deps/share deps/tmp deps/.stamp-*
	@rm -rf bin
	@rm -rf build
	@rm -rf tmp
	@echo "[CLEAN] Clean ."
