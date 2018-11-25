#!/bin/bash

# for all years between 1998 and 2016
# download the daily files using the corresponding download link file
# for YEAR in {1998..2016}
# do
wget \
	--load-cookies ~/.urs_cookies \
	--save-cookies ~/.urs_cookies \
	--auth-no-challenge=on \
	--keep-session-cookies \
	--content-disposition \
	--user=rschlaefli \
	--ask-password \
	-i ./overall_urls_v3.txt
	# -i ./overall_urls.txt
	# -i ./${YEAR}_download_urls.inp
# done

# cleanup filename of all downloaded files
# cuts of clutter in the end (url parameters)
rename 's/.7.nc4.nc4.*/_v3.trmm/' ./*
