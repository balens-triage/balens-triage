# BugTriaging benchmarker

## Install 

## configuration

## database

Have a postgresql database running on port 5432 e.g. using a docker container like this:

```
docker run --name psql -d -p 5432:5432 --rm -e POSTGRES_USER=cp -e POSTGRES_PASSWORD=PWD -e POSTGRES_DB=pbtr -v pgdata:/var/lib/postgresql/data postgres
```

If you don't want the database to be accessible from the web, change the port definition to:

`-p 127.0.0.1:5432:5432`

        
## setting up the stackoverflow data

In order to set up the stackoverflow data we need to download their public database dump from 2014 here:

https://archive.org/details/stackexchange

We did so using the torrent. Then we need to extract the relevant data using 7z decompression. For this
we used the following script so that they end up in the `stackoverflow/` folder.

````bash
for f in $(ls stackexchange | grep stackoverflow.com-);
do
    7z x -aoa -o./stackoverflow "./stackoverflow/$f";
done
````

The rest of the process is automated by calling the main script like this:

````bash
python run.py --stackoverflow
````
