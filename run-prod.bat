@echo off
echo ========================================
echo   Consumption Forecast - Production
echo ========================================
echo.

:: Stop existing container if running
docker stop consumption-forecast-app 2>nul
docker rm consumption-forecast-app 2>nul

echo Building latest image...
docker build -t consumption-forecast .

echo.
echo Starting production container...
docker run -d ^
  -p 8501:8501 ^
  -v "%cd%\data:/app/data:ro" ^
  --name consumption-forecast-app ^
  consumption-forecast

echo.
echo ========================================
echo   App is running at: http://localhost:8501
echo ========================================
echo.
echo To view logs: docker logs -f consumption-forecast-app
echo To stop app:  docker stop consumption-forecast-app