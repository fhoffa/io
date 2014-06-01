#!/bin/bash -e

# https://console.developers.google.com/project/apps~cloude-sandbox/storage/toque-demo/
BUCKET=toque-demo

# Enable Allow-Origin: * headers on the demo data bucket
# so that data can be pulled into a webapp via an AJAX 
# request.
# https://developers.google.com/storage/docs/cross-origin
gsutil cors set cors.xml gs://$BUCKET

# Make the demo data world-readable so that a browser can
# grab it. This only contains aggregate PDFs, not detailed touch data.
gsutil -m acl set -R -a public-read gs://$BUCKET

# It will take a while for Allow-Origin changes to propagate
# to existing objects since things are aggresively cached.

