#!/bin/bash
cd $(dirname $0)

if [ ! -n "$SCHRODINGER" ]
then
        echo "SCHRODINGER not defined"
        exit
fi

if [ ! -f ../grid/$(cat grid.info).zip ]
then
        echo "$(cat grid.info) not exist"
        exit
fi

for ((i=0;i<6;i++))
do
        if [ ! -d run_$i ]
        then
                mkdir run_$i
                touch run_$i/running
                workdir=run_$i
                break
        elif [ ! -f run_$i/running ]
        then
                rm -r run_$i
                mkdir run_$i
                touch run_$i/running
                workdir=run_$i
                break
        fi
done

if [ ! -n "$workdir" ]
then
        echo "thread Full!"
        exit
fi

echo -e "n\n1" > output.csv
cp vsw.inp $workdir/vsw_run.inp
cp ligand.smi $workdir/ligand.smi
cp grid.info $workdir/grid.info
cp ../grid/$(cat grid.info).zip $workdir/grid.zip
cd $workdir
sed -i "s#root_path#$(pwd)#g" vsw_run.inp
"$SCHRODINGER/vsw" vsw_run.inp -OVERWRITE -host_ligprep localhost:8 -host_glide localhost:8 -adjust -NJOBS 8 -TMPLAUNCHDIR > run.info
jobid="$(sed -n '$ s/.*JobId: \(.*\)\s*/\1/p' run.info)"
echo "jobid: $jobid, waiting the job to finish."
"$SCHRODINGER/jobcontrol" -wait $jobid
if [ -f vsw_run-HTVS_OUT_csv-001.csv ]
then
    cp vsw_run-HTVS_OUT_csv-001.csv ../output.csv
else
    echo "no docking pose"
fi

rm running vsw_run-HTVS_OUT_csv-001.csv
