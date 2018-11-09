pushd data || exit


wget https://www.dropbox.com/s/jtqnvzg3jz6dctq/stimuli_384sentences.txt?dl=1
wget https://www.dropbox.com/s/r9gi8p48jva3b4w/stimuli_384sentences_dereferencedpronouns.txt?dl=1
wget https://www.dropbox.com/s/10jhiaievf07b93/vectors_384sentences.GV42B300.average.txt?dl=1
wget https://www.dropbox.com/s/10jhiaievf07b93/vectors_384sentences_dereferencedpronouns.GV42B300.average.txt?dl=1

wget https://www.dropbox.com/s/bdll04a2h4ou4xj/P01.tar?dl=1
wget https://www.dropbox.com/s/wetd2gqljfbh8cg/M02.tar?dl=1
wget https://www.dropbox.com/s/b7tvvkrhs5g3blc/M04.tar?dl=1
wget https://www.dropbox.com/s/izwr74rxn637ilm/M07.tar?dl=1
wget https://www.dropbox.com/s/3q6xhtmj611ibmo/M08.tar?dl=1
wget https://www.dropbox.com/s/kv1wm2ovvejt9pg/M09.tar?dl=1
wget https://www.dropbox.com/s/2h6kmootoruwz52/M14.tar?dl=1
wget https://www.dropbox.com/s/u19wdpohr5pzohr/M15.tar?dl=1

find . -name '*\?*' | while read -r path ; do
    mv "$path" "${path%\?*}"
done

popd
