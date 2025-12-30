.PHONY: clean
clean:
	rm -rf images/*
inference_openvino:
	./run_openvino.sh -m models/yolox_s.onnx -i test_images/street_scene.png
	