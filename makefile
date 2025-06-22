.PHONY: test clean

kill_docker:
	osascript -e 'quit app "Docker"'


docker:
	open -a Docker
	docker build -t micrograd:test .

test:
	python3 -m pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.egg-info" -delete
	rm -rf build
