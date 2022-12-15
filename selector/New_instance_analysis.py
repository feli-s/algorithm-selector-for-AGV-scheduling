import numpy as np
import pandas as pd
import os


def read_benchmark_instance_JSSP(filename):
    name = os.path.basename(filename)
    name = name.replace('.txt', '')

    jobs = []
    operations = []

    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[0].split()
    #Number of jobs
    nb_jobs = int(first_line[0])
    #Number of AGVs
    nb_AGVs = int(first_line[1])

    with open(filename) as file:
        for line in file.readlines()[1:]:
            strip_lines = line.strip()
            jobs_list = strip_lines.split()
            jobs_list = list(map(int, jobs_list))
            jobs.append(jobs_list)

        tasklist = []
        for operation in jobs:
            tuple = [(operation[i], operation[i + 1]) for i in range(0, len(operation), 2)]
            tasklist.append(tuple)

        operations.append(tasklist)

    x = operations[0][0]
    nb_operations_list = []
    for j in x:
        length = len(x)
        nb_operations_list.append(length)

    nb_operations = sum(nb_operations_list)

    if nb_jobs != len(jobs):
        nb_jobs = len(jobs)

    #Job-AGV ratio
    job_agv_ratio = nb_jobs/nb_AGVs
    #skewness
    skewness = max(nb_jobs/nb_AGVs, nb_AGVs/nb_jobs)

    # Transport times for each job on each AGV (given in the processing order)
    transport_times = []
    for t in jobs:
        times_in_job = []
        for index in range(1, len(t), 2):
            times_in_job.append(t[index])
        transport_times.append(times_in_job)

    sum_job_times = []
    for v in transport_times:
        job_sum = sum(v)
        sum_job_times.append(job_sum)

    max_job_duration = max(sum_job_times)
    min_job_duration = min(sum_job_times)

    #Get single transport times
    travel_times = []
    for item in transport_times:
        for time in item:
            travel_times.append(time)

    max_dur_operation = max(travel_times)
    min_dur_operation = min(travel_times)


    #mean of operation times
    mean_per_job = []
    for value in transport_times:
        operation = np.mean(value)
        mean_per_job.append(operation)
    #print(mean_per_job)
    #total_job_mean = np.mean(mean_per_job)
    mean_operation_duration_per_job = np.mean(mean_per_job)

    data = {'Instance': [name],
            'jobs': [nb_jobs],
            'AGVs': [nb_AGVs],
            'Operations': [nb_operations],
            'Job_AGV_ratio': [job_agv_ratio],
            'Skewness': [skewness],
            'Mean operation duration per job': [mean_operation_duration_per_job],
            'Max job duration': [max_job_duration],
            'Min job duration': [min_job_duration],
            'Max_operation_duration': [max_dur_operation],
            'Min_operation_duration': [min_dur_operation]}

    df = pd.DataFrame(data)

    return df

def analyse_instance_FJSSP(instance):
    name = os.path.basename(instance)
    name = name.replace('.fjs', '')

    jobs = []
    operations = []

    with open(instance) as f:
        lines = f.readlines()

    first_line = lines[0].split()
    # Number of jobs
    nb_jobs = int(first_line[0])
    # Number of AGVs
    nb_AGVs = int(first_line[1])

    with open(instance) as file:
        for line in file.readlines()[1:]:
            strip_lines = line.strip()
            jobs_list = strip_lines.split()
            jobs_list = list(map(int, jobs_list))
            jobs.append(jobs_list)

        for operation in jobs:
            nb_tasks = len(operation)
            tasklist = []
            n = 1
            while n < nb_tasks:
                k = operation[n]  # 1
                f = n + 1
                tuple = [(operation[f], operation[f + 1]) for f in range(f, n+k*2, 2)]
                tasklist.append(tuple)
                n = (2*k + n) + 1
            operations.append(tasklist)

    nb_operations_list = []
    count = 0
    for j in operations:
        length = operations[count]
        leOp = len(length)
        nb_operations_list.append(leOp)
        count = count+1

    nb_operations = sum(nb_operations_list)

    if nb_jobs != len(jobs):
        nb_jobs = len(jobs)

    #Job-AGV ratio
    job_agv_ratio = nb_jobs/nb_AGVs
    #skewness
    skewness = max(nb_jobs/nb_AGVs, nb_AGVs/nb_jobs)

    operation_times = []
    job_times = []
    horizon = 0
    for job in operations:
        totalJob = []
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[1])
            horizon += max_task_duration
            totalJob.append(max_task_duration)
            operation_times.append(max_task_duration)
        job_times.append(totalJob)

    #print('Horizon = %i' % horizon)

    max_operation_duration = max(operation_times)
    min_operation_duration = min(operation_times)

    operation_mean_per_job = []
    for value in job_times:
        oper = np.mean(value)
        operation_mean_per_job.append(oper)

    total_job_times = []
    for value in job_times:
        job_sum = sum(value)
        total_job_times.append(job_sum)

    max_job_duration = max(total_job_times)
    min_job_duration = min(total_job_times)

    total_job_mean = sum(operation_mean_per_job)

    data = {'instance': [name],
            'jobs': [nb_jobs],
            'AGVs': [nb_AGVs],
            'Operations': [nb_operations],
            'Job_AGV_ratio': [job_agv_ratio],
            'Skewness': [skewness],
            'Total_job_mean': [total_job_mean],
            'Max_transport_duration': [max_operation_duration],
            'Min_transport_duration': [min_operation_duration],
            'Max_job_duration': [max_job_duration],
            'Min_job_duration': [min_job_duration],
            'Horizon': [horizon]}

    df = pd.DataFrame(data)

    return df


