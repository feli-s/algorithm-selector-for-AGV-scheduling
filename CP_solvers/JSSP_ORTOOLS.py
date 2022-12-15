
import collections
import os

import pandas as pd
from ortools.sat.python import cp_model

def read_instance_JSSP(filename):

    with open(filename) as f:
        lines = f.readlines()
    name = os.path.basename(filename)
    name = name.replace('.txt', '')

    jobs = []
    job_data = []
    with open(filename) as file:
        for line in file.readlines()[1:]:
            strip_lines = line.strip()
            jobs_list = strip_lines.split()
            jobs_list = list(map(int, jobs_list))
            m = jobs.append(jobs_list)

    for operation in jobs:
        tuples = [(operation[i], operation[i+1]) for i in range(0, len(operation),2)]
        n = job_data.append(tuples)

    return job_data, name

def ORTOOLS_JSSP(data):
    """OR TOOLS."""
    # Data.
    jobs_data = data
    print('Read instance data..')

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()

    #Set time limit
    solver.parameters.max_time_in_seconds = 10.0

    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Solution:')
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(start=solver.Value(
                        all_tasks[job_id, task_id].start),
                                       job=job_id,
                                       index=task_id,
                                       duration=task[1]))

        # Create per machine output lines.
        output = ''
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_task_%i' % (assigned_task.job,
                                           assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-15s' % name

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-15s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f'Optimal Schedule Length: {solver.ObjectiveValue()}')
        print(output)
    else:
        print('No solution found.')

    # Statistics.
    print('\nStatistics')
    print('  - conflicts: %i' % solver.NumConflicts())
    print('  - branches : %i' % solver.NumBranches())
    print('  - wall time: %f s' % solver.WallTime())

    makespan = solver.ObjectiveValue()
    solving_time = solver.WallTime()

    return makespan, solving_time


path_of_directories = r'C:\[..]'
cols = ['instance', 'makespan', 'solving time', 'solver']
files = []
lst = []
for names in os.listdir(path_of_directories):
    f = os.path.join(path_of_directories, names)
    files.append(f)
    if os.path.isfile(f):
        data, name = read_instance_JSSP(f)
        makespan, solving_time = ORTOOLS_JSSP(data)
        solver = 'ORTOOLS'
        lst.append([name, makespan, solving_time, solver])

df1 = pd.DataFrame(lst, columns=cols)
inst_analysis = df1.to_excel('instance_analysis_ORTOOLS_JSSP.xlsx')


