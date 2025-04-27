# Crawl Data

filename = 'dataset 14-4-2025.csv'
search_keyword = 'palestine lang:en until:2025-4-7 since:2024-10-7'
limit = 5000

!npx --yes tweet-harvest@latest -o "{filename}" -s "{search_keyword}" -l {limit} --token ""