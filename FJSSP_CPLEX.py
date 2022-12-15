import collections
import datetime

from docplex.cp.model import *

import plotly as py
import plotly.figure_factory as ff
dt = datetime.datetime
time_delta = datetime.timedelta
pyplt = py.offline.plot
import random
import os
import pandas as pd

def rgb():
    return random.randint(0, 256)

def read_instance_FJSSP(instance):
    jobs = []
    operations = []

    with open(instance) as f:
        lines = f.readlines()
    name = os.path.basename(instance)
    name = name.replace('.txt', '')

    first_line = lines[0].split()
    # Number of jobs
    nb_jobs = int(first_line[0])
    #Number of AGVs
    nb_AGVs = int(first_line[1])


    with open(instance) as file:
        for line in file.readlines()[1:]:
            strip_lines = line.strip()
            jobs_list = strip_lines.split()
            jobs_list = list(map(int, jobs_list))
            jobs.append(jobs_list)

        for operation in jobs:
            #TODO: different task calculation (not first line, maybe delete first line and just count len(operation)
            #nb_tasks = operation[0]
            nb_tasks = len(operation)
            tasklist = []
            n = 1
            while n < nb_tasks:
            #for i in range(1, nb_tasks+1):
           #for i in range(1, nb_tasks-1):
                k = operation[n]  # 1
                f = n + 1
                tuple = [(operation[f], operation[f + 1]) for f in range(f, n+k*2, 2)]
                tasklist.append(tuple)
                n = (2*k + n) + 1
            operations.append(tasklist)

    return operations, nb_AGVs, name, nb_jobs


def flexible_jobshop_CPLEX(data, AGVs, name, nb_jobs):
    """Solve a small flexible jobshop problem."""
    # Data part.
    jobs = data
    instance = name
    nb_AGVs = AGVs
    nb_jobs = nb_jobs

    num_jobs = len(jobs)
    all_jobs = range(num_jobs)
    num_machines = AGVs
    all_machines = range(num_machines)
    machines_list = [*range(0,num_machines+1)]
    jobs_list = [*range(0, num_jobs)]


    # create DataFrame that contains number of operations per job
    JobsOperationsTable = pd.DataFrame(columns=['op_id', 'job', 'priority'])
    count = 0
    opid = 0
    priority = []
    oper_id = []
    joblist = []
    nbr_operations = len(jobs[count])
    for j in jobs:
        for o in range(1, nbr_operations+1):
            ope_id = ('opr_'+str(opid)+'_'+str(o))
            oper_id.append(ope_id)
            j = count
            prio = o
            joblist.append(j)
            priority.append(prio)
        opid = opid + 1
        count = count + 1
    #print(JobsOperationsTable)
    JobsOperationsTable["op_id"] = oper_id
    JobsOperationsTable["job"] = joblist
    JobsOperationsTable["priority"] = priority
    from collections import namedtuple
    TJobsOperations = namedtuple("TJobsOperations", ["op_id", "job", "priority"])
    JobsOperations = [TJobsOperations(*joboperations_row) for joboperations_row in JobsOperationsTable.itertuples(index=False)]
    print(JobsOperations)

    #Create DataFrame that contains machine id and processing time
    OperationMachinesTable = pd.DataFrame(columns=['op_id', 'machine', 'proc_time'])
    count = 0
    opid = 0
    nbr_operations = len(jobs[count])
    machine_id = []
    processing_time = []
    operation_id = []
    for j in jobs:
        t = 1
        for task in j:
            l = len(task)
            for a in range (0,l):
                machine1 = task[a][0]
                time = task[a][1]
                op_id = ('opr_'+str(count)+'_'+str(t))
                machine_id.append(machine1)
                processing_time.append(time)
                operation_id.append(op_id)
            t = t+1
        count = count+1

    OperationMachinesTable["op_id"] = operation_id
    OperationMachinesTable["machine"] = machine_id
    OperationMachinesTable["proc_time"] = processing_time

    TOperationMachines = namedtuple("TOperationMachines",['op_id', 'machine', 'proc_time'])
    OperationMachines = [TOperationMachines(*operationmachines_row) for operationmachines_row in
                         OperationMachinesTable.itertuples(index=False)]
    print(OperationMachines)

    mdl = CpoModel(name='parallelMachineScheduling_FlexibleJobShop')

    # define interval variables
    jobops_itv_vars = {}
    for jo in JobsOperations:
        jobops_itv_vars[(jo.op_id, jo.job, jo.priority)] = mdl.interval_var(
            name="operation {} job {} priority {}".format(jo.op_id, jo.job, jo.priority))

    opsmchs_itv_vars = {}
    for om in OperationMachines:
        opsmchs_itv_vars[(om.op_id, om.machine)] = mdl.interval_var(optional=True, size=om.proc_time,
                                                                    name="operation {} machine {}".format(om.op_id,
                                                                                                          om.machine))

    # minimize makespan
    objective = mdl.max([mdl.end_of(opsmchs_itv_vars[(op_id, machine)]) for (op_id, machine) in opsmchs_itv_vars])
    mdl.add(mdl.minimize(objective))

    # Force no overlap for operations executed on an equal machine
    machine_operations = [[] for m in machines_list]
    for (op_id, machine) in opsmchs_itv_vars:
        machine_operations[machine].append(opsmchs_itv_vars[(op_id, machine)])
    if machine_operations[0] == []:
        del machine_operations[0]
    for mops in machine_operations:
        mdl.add(mdl.no_overlap(mops))

    # Each operation must start after the end of the predecessor
    previuosops=dict()
    for jo1 in JobsOperations:
        for jo2 in JobsOperations:
            if jo1.job==jo2.job and jo1.priority+1==jo2.priority:
                previuosops[jo2]=jo1.op_id
    for j in jobs_list:
        for jo in JobsOperations:
            if jo.job==j and jo.priority>=2:
                mdl.add(mdl.end_before_start(jobops_itv_vars[(previuosops[jo],jo.job, jo.priority-1)], jobops_itv_vars[(jo.op_id,jo.job, jo.priority)]))

    # job operation intervals can only take value if one of alternative operation-machines intervals take value
    for (op_id, job, priority) in jobops_itv_vars:
        mdl.add(mdl.alternative(jobops_itv_vars[(op_id, job, priority)],
                                [opsmchs_itv_vars[(o, m)] for (o, m) in opsmchs_itv_vars if (o == op_id)], 1))

    msol = mdl.solve(log_output=True, TimeLimit=10)

    makespan = msol.get_objective_value()
    #print(makespan)
    #solve_time = docplex.cp.solution.CpoSolveResult.get_solve_time(res)
    solve_time = msol.get_solve_time()

    print("Solution: ")
    msol.print_solution()

    import docplex.cp.utils_visu as visu
    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = 25, 5

    '''if msol and visu.is_visu_enabled():
        visu.timeline("Solution Schedule", 0, 100)
        for m in machines_list:
            visu.sequence(name='M' + str(m))
            for (op_id, job, priority) in jobops_itv_vars:
                for (o, mch) in opsmchs_itv_vars:
                    itv2 = msol.get_var_solution(opsmchs_itv_vars[(o, mch)])
                    if op_id == o and m == mch and itv2.is_present():
                        itv = msol.get_var_solution(jobops_itv_vars[(op_id, job, priority)])
                        visu.interval(itv, m, 'J' + str(job) + '_' + str(op_id))
            visu.show()'''


    return instance, makespan, solve_time


#instance = "FJSSP1test.fjs"
#data, nb_AGVs, name, nb_jobs = read_instance_FJSSP(instance)
#instance, makespan, solve_time = flexible_jobshop_CPLEX(data, nb_AGVs, name, nb_jobs)

path_of_directories = r'C:\Users\felis\Coding\ML_SELECTOR_ALGORITHMS\Instances\Job_Data\Barnes\Text'
cols = ['instance', 'makespan', 'solving time', 'solver']
files = []
lst = []
for names in os.listdir(path_of_directories):
    f = os.path.join(path_of_directories, names)
    files.append(f)
    if os.path.isfile(f):
        data, nb_AGVs, name, nb_jobs = read_instance_FJSSP(f)
        instance, makespan, solving_time = flexible_jobshop_CPLEX(data, nb_AGVs, name, nb_jobs)
        solver = 'CPLEX'
        lst.append([instance, makespan, solving_time, solver])

df1 = pd.DataFrame(lst, columns=cols)
inst_analysis = df1.to_excel('instance_analysis_CPLEX_Barnes.xlsx')


