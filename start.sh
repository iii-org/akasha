#!/bin/bash
uvicorn api:app --workers 4 --host 0.0.0.0 --port 8000
