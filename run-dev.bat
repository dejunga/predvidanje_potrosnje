@echo off
echo ========================================
echo   Consumption Forecast - Development
echo ========================================
echo.

:: Stop existing container if running
docker stop consumption-forecast-dev 2>nul
docker rm consumption-forecast-dev 2>nul

echo Starting development container with hot reloading...
echo App will be available at: http://localhost:8502
echo.
echo Changes to your code will automatically reload the app!
echo Press Ctrl+C to stop the development server.
echo.

:: Run development container with volume mounting for hot reloading
docker run -it --rm ^
  -p 8502:8501 ^
  -v "%cd%:/app" ^
  -v "/app/__pycache__" ^
  -e STREAMLIT_SERVER_RUN_ON_SAVE=true ^
  -e STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll ^
  --name consumption-forecast-dev ^
  consumption-forecast ^
  streamlit run app_enhanced.py ^
  --server.port=8501 ^
  --server.address=0.0.0.0 ^
  --server.headless=true ^
  --server.enableCORS=false ^
  --server.enableXsrfProtection=false ^
  --server.runOnSave=true ^
  --server.fileWatcherType=poll