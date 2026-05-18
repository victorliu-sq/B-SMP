.PHONY: clean

clean:
	@echo "[CLEAN] Clean Begins."
	@echo "[CLEAN] Remove $(CURDIR)/third_party ..."
	@rm -rf third_party
	@echo "[CLEAN] Remove $(CURDIR)/bin ..."
	@rm -rf bin
	@echo "[CLEAN] Remove $(CURDIR)/build ..."
	@rm -rf build
	@echo "[CLEAN] Clean ."
