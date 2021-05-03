mkdir -p ~/.streamlit/

# https://docs.streamlit.io/en/stable/streamlit_configuration.html

echo "\
[general]\n\
email = \"dummy@email.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[global]\n\
disableWatchdogWarning = false\n\
sharingMode = \"off\"\n\
[server]\n\
headless = true\n\
enableCORS=false\n\
folderWatchBlacklist = ['*']\n\
port = $PORT\n\
[theme]
primaryColor =  \"#F7630C\"\n\
textColor=\"#EFECEC\" \n\
backgroundColor = \"#525050\"\n\
secondaryBackgroundColor=\"#2E2D2D\"\n\
" > ~/.streamlit/config.toml
