#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=1G
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH --output=results/resubmission-%j.out

user="leyizh"
submission_record="results/submission_record.csv"
submission_record_new="results/submission_record_new.csv"
resubmission_record="results/resubmission_record.csv"

num_jobs="$(($(wc -l < $submission_record) - 1))"
echo "total number of jobs:" $num_jobs

num_pending="$(squeue -u $user -t PD -h | wc -l)"
num_running="$(squeue -u $user -t R -h | wc -l)"
echo "number of jobs pending:" $num_pending
echo "number of jobs running" $num_running
if [[ $((num_pending + num_running)) > 0 ]]; then 
	echo "some jobs still unfinished"
fi

echo "parameter_id,realization,job_id,seed,previous_status" > $submission_record_new
echo "parameter_id,realization,job_id,seed,job_status,request_old,request_new,job_id_new" > $resubmission_record

count_CD=0
count_OOM=0
count_TO=0
count_other=0

for ((i=2; i<=($num_jobs+1); i++)); do
    parameter_id="$(awk -v i="$i" -F',' 'NR==i{print $1}' $submission_record)"
    realization="$(awk -v i="$i" -F',' 'NR==i{print $2}' $submission_record)"
    job="$(awk -v i="$i" -F',' 'NR==i{print $3}' $submission_record)"
    seed="$(awk -v i="$i" -F',' 'NR==i{print $4}' $submission_record | tr -d '\r')"

    # return job status
    job_status="$(sacct --jobs=$job --format=state --noheader | head -n 1)"

    if [ $job_status == "COMPLETED" ]; then
	((count_CD+=1))
	# record the old job info in file since nothing changed
	echo $parameter_id,$realization,$job,$seed,$job_status >> $submission_record_new
    elif [ $job_status == "OUT_OF_ME+" ]; then
	((count_OOM+=1))
	# find old memory request
	mem_old="$(sacct --jobs=$job --format=ReqMem --noheader | head -n 1)"
    	mem_unit="$(echo $mem_old | grep -oE "[A-Za-z]")"
	mem_old_num="$(echo $mem_old | grep -oE '[0-9]+')"
	# compute new memory request
	mem_new_num=$((2 * mem_old_num))
    	mem_new="$mem_new_num$mem_unit"
	# submit job with new momory request
	job_name=p$parameter_id-r$realization
	job_submit="$(sbatch --mem=$mem_new --job-name=$job_name job_submit_template.sh $parameter_id $realization $seed)"
	job_new="$(echo "${job_submit}" | grep -oE '[0-9]+')"
	# record new job id and info
	echo $parameter_id,$realization,$job_new,$seed,$job_status >> $submission_record_new
	echo $parameter_id,$realization,$job, $seed,$job_status,$mem_old,$mem_new,$job_new >> $resubmission_record
    elif [ $job_status == "TIMEOUT" ]; then
	((count_TO+=1))
	# fine old time limit request
	time_old="$(sacct --jobs=$job --format=Timelimit --noheader | head -n 1)"
	time_old_sec="$(echo $time_old | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')"
	# compute new time limit request
	time_new_sec="$((2 * time_old_sec))"
	time_new="$(date -d@$time_new_sec -u +%H:%M:%S)"
	# submit job with new time request
	job_name=p$parameter_id-r$realization
	job_submit="$(sbatch --time=$time_new --job-name=$job_name job_submit_template.sh $parameter_id $realization $seed)"
	job_new="$(echo "${job_submit}" | grep -oE '[0-9]+')"
	# record new job id and info
	echo $parameter_id,$realization,$job_new,$seed,$job_status >> $submission_record_new
	echo $parameter_id,$realization,$job, $seed,$job_status,$time_old,$time_new,$job_new >> $resubmission_record
    else
	((count_other+=1))
	echo "investigate:" $parameter_id, $realization, $job, $seed, $job_status
	echo $parameter_id,$realization,$job,$seed,$job_status >> $submission_record_new
    fi
done

echo "total complete:" $count_CD
echo "total out of memory:" $count_OOM
echo "total timeout:" $count_TO
echo "total other:" $count_other


