import docplex.cp.solution
from docplex.cp.model import *
import os

import pandas as pd

def read_instance(filename):

    with open(filename) as f:
        lines = f.readlines()
    instance = os.path.basename(filename)
    instance = instance.replace('.txt', '')

    with open(filename, 'r') as file:
        NB_JOBS, NB_MACHINES = [int(v) for v in file.readline().split()]
        JOBS = [[int(v) for v in file.readline().split()] for i in range(NB_JOBS)]

    return NB_JOBS, NB_MACHINES, JOBS, instance

def CPLEX_JSSP(NB_JOBS, NB_MACHINES, JOBS):
    NB_MACHINES = NB_MACHINES
    JOBS = JOBS
    NB_JOBS = NB_JOBS

    # Build list of machines. MACHINES[j][s] = id of the machine for the operation s of the job j
    MACHINES = [[JOBS[j][2 * s] for s in range(NB_MACHINES)] for j in range(NB_JOBS)]

    # Build list of durations. DURATION[j][s] = duration of the operation s of the job j
    DURATION = [[JOBS[j][2 * s + 1] for s in range(NB_MACHINES)] for j in range(NB_JOBS)]

    # Create model
    mdl = CpoModel()

    # Create one interval variable per job operation
    job_operations = [[interval_var(size=DURATION[j][m], name='O{}-{}'.format(j,m)) for m in range(NB_MACHINES)]
                      for j in range(NB_JOBS)]

    # Each operation must start after the end of the previous
    for j in range(NB_JOBS):
        for s in range(1, NB_MACHINES):
            mdl.add(end_before_start(job_operations[j][s-1], job_operations[j][s]))

    # Force no overlap for operations executed on a same machine
    machine_operations = [[] for m in range(NB_MACHINES)]
    for j in range(NB_JOBS):
        for s in range(NB_MACHINES):
            machine_operations[MACHINES[j][s]].append(job_operations[j][s])
    for mops in machine_operations:
        mdl.add(no_overlap(mops))

    # Minimize termination date
    mdl.add(minimize(max(end_of(job_operations[i][NB_MACHINES-1]) for i in range(NB_JOBS))))

    # Solve model
    print('Solving model...')
    res = mdl.solve(TimeLimit=10)
    print('Solution:')
    res.print_solution()

    makespan = res.get_objective_value()
    #print(makespan)
    #solve_time = docplex.cp.solution.CpoSolveResult.get_solve_time(res)
    solve_time = res.get_solve_time()
    #print(solve_time)

    return instance, makespan, solve_time

    # Draw solution
    '''import docplex.cp.utils_visu as visu
    if res and visu.is_visu_enabled():
        visu.timeline('Solution for job-shop ' + filename)
        visu.panel('Jobs')
        for i in range(NB_JOBS):
            visu.sequence(name='J' + str(i),
                          intervals=[(res.get_var_solution(job_operations[i][j]), MACHINES[i][j], 'M' + str(MACHINES[i][j])) for j in
                                     range(NB_MACHINES)])
        visu.panel('Machines')
        for k in range(NB_MACHINES):
            visu.sequence(name='M' + str(k),
                          intervals=[(res.get_var_solution(machine_operations[k][i]), k, 'J' + str(i)) for i in range(NB_JOBS)])
        visu.show()'''


path_of_directories = r'C:\[..]'
cols = ['instance', 'makespan', 'solving time', 'solver']
files = []
lst = []
for names in os.listdir(path_of_directories):
    f = os.path.join(path_of_directories, names)
    files.append(f)
    if os.path.isfile(f):
        NB_JOBS, NB_MACHINES, JOBS, instance = read_instance(f)
        instance, makespan, solving_time = CPLEX_JSSP(NB_JOBS, NB_MACHINES, JOBS)
        solver = 'CPLEX'
        lst.append([instance, makespan, solving_time, solver])

df1 = pd.DataFrame(lst, columns=cols)
inst_analysis = df1.to_excel('instance_analysis_CPLEX_JSSP.xlsx')

