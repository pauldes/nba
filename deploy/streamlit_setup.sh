mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"dummy@email.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\

[global]\
disableWatchdogWarning = false\
sharingMode = "off"\
[server]\n\
headless = true\n\
enableCORS=false\n\
watchFileSystem = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
