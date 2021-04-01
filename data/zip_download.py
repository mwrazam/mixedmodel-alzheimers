import requests
def download_zip(zip_url,file_name): # tested works ok
    r = requests.get(zip_url, stream=True)
    # Stream: A Boolean indication if the response
    # should be immediately downloaded (False) or streamed (True).
    open(file_name,'wb').write(r.content)

for i in range(12): # just test one file for now
    zip_number = str(i+1)
    print(zip_number)
    zip_url = 'https://download.nrg.wustl.edu/data/oasis_cross-sectional_disc'+zip_number+'.tar.gz'
    print('zip_url is ', zip_url)
    file_name = zip_url.rsplit('/', 1)[1]
    print('file_name is ', file_name)
    download_zip(zip_url, file_name)
