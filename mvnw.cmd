@REM ----------------------------------------------------------------------------
@REM Maven Start Up Batch script
@REM ----------------------------------------------------------------------------
@echo off
setlocal
set "MVNW_REPOURL=https://repo.maven.apache.org/maven2"
set "MAVEN_VERSION=3.9.6"

set "WRAPPER_JAR=.mvn\wrapper\maven-wrapper.jar"
set "WRAPPER_PROPERTIES=.mvn\wrapper\maven-wrapper.properties"

if exist "%WRAPPER_JAR%" (
    goto :run
)

echo "Maven Wrapper not found, downloading..."
mkdir .mvn\wrapper 2>NUL
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('%MVNW_REPOURL%/org/apache/maven/wrapper/maven-wrapper/3.2.0/maven-wrapper-3.2.0.jar', '%WRAPPER_JAR%') }"

:run
java -Dmaven.multiModuleProjectDirectory="%~dp0" -jar "%WRAPPER_JAR%" %*