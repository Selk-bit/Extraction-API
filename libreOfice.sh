#!/usr/bin/env bash

STORAGE_DIR=/opt/render/project/.render

# Create a directory for LibreOffice
mkdir -p $STORAGE_DIR/libreoffice

# Navigate to the directory
cd $STORAGE_DIR/libreoffice

# Download LibreOffice tarball (replace with the desired version)
wget https://download.documentfoundation.org/libreoffice/stable/24.2.7/deb/x86_64/LibreOffice_24.2.7_Linux_x86-64_deb.tar.gz

# Extract the tarball
tar -xzf LibreOffice_24.2.7_Linux_x86-64_deb.tar.gz

# Install the DEB packages
sudo dpkg -i LibreOffice_24.2.7.2_Linux_x86-64_deb/DEBS/*.deb

# Cleanup
rm -rf LibreOffice_24.2.7_Linux_x86-64_deb.tar.gz LibreOffice_24.2.7.2_Linux_x86-64_deb

cd $HOME/project/src 