"""
Solves a flexible jobshop problems with the CP-SAT solver.
"""
import collections
import datetime

from ortools.sat.python import cp_model

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

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        """Called at each new solution."""
        print('Solution %i, time = %f s, objective = %i' %
              (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
        self.__solution_count += 1

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

    return operations, nb_AGVs, name


def flexible_jobshop(data, AGVs, name):
    # Data part.
    jobs = data
    instance = name
    print(instance)

    num_jobs = len(jobs)
    all_jobs = range(num_jobs)
    num_machines = AGVs
    all_machines = range(num_machines)

    # Model the flexible jobshop problem.
    model = cp_model.CpModel()

    horizon = 0
    for job in jobs:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[1])
            horizon += max_task_duration

    print('Horizon = %i' % horizon)

    # Global storage of variables.
    intervals_per_resources = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []
    task_type = collections.namedtuple("task_type", "start end interval")
    assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration")

    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        job = jobs[job_id]
        num_tasks = len(job)
        previous_end = None
        for task_id in range(num_tasks):
            task = job[task_id]

            min_duration = task[0][1]
            max_duration = task[0][1]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][1]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = '_j%i_t%i' % (job_id, task_id)
            start = model.NewIntVar(0, horizon, 'start' + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration,
                                       'duration' + suffix_name)
            end = model.NewIntVar(0, horizon, 'end' + suffix_name)
            interval = model.NewIntervalVar(start, duration, end,
                                            'interval' + suffix_name)

            # Store the start for the solution.
            starts[(job_id, task_id)] = start

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = '_j%i_t%i_a%i' % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar('presence' + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, 'start' + alt_suffix)
                    l_duration = task[alt_id][1]
                    l_end = model.NewIntVar(0, horizon, 'end' + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence,
                        'interval' + alt_suffix)
                    l_presences.append(l_presence)

                    # Link the primary/global variables with the local ones.
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    # Add the local interval to the right machine.
                    intervals_per_resources[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution.
                    presences[(job_id, task_id, alt_id)] = l_presence

                # Select exactly one presence variable.
                model.AddExactlyOne(l_presences)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)

        job_ends.append(previous_end)

    # Create machines constraints.
    for machine_id in all_machines:
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    # Solve model.
    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter()

    #Set time limit of 10s
    solver.parameters.max_time_in_seconds = 10.0

    status = solver.Solve(model, solution_printer)

    # Print final solution.
    assigned_jobs_fjssp = collections.defaultdict(list)
    for job_id in all_jobs:
        print('Job %i:' % job_id)
        for task_id in range(len(jobs[job_id])):
            start_value = solver.Value(starts[(job_id, task_id)])
            machine = -1
            duration = -1
            selected = -1
            for alt_id in range(len(jobs[job_id][task_id])):
                if solver.Value(presences[(job_id, task_id, alt_id)]):
                    duration = jobs[job_id][task_id][alt_id][1]
                    machine = jobs[job_id][task_id][alt_id][0]
                    selected = alt_id
                    assigned_jobs_fjssp[machine].append(
                        assigned_task_type(start=start_value, job=job_id, index=task_id, duration=duration))

            print(
                '  task_%i_%i starts at %i (alt %i, machine %i, duration %i)' %
                (job_id, task_id, start_value, selected, machine, duration))

    obj_makespan = solver.ObjectiveValue()
    solving_time = solver.WallTime()
    print('Solve status: %s' % solver.StatusName(status))
    print('Optimal objective value: %i' % solver.ObjectiveValue())
    print('Statistics')
    print('  - conflicts : %i' % solver.NumConflicts())
    print('  - branches  : %i' % solver.NumBranches())
    print('  - wall time : %f s' % solver.WallTime())
    
    return instance, obj_makespan, solving_time

def draw_ganttchart(all_machines, assigned_jobs_fjssp, num_jobs):
    dfg = []
    today = dt.today()
    start_date = dt(today.year, today.month, today.day)

    for machine in all_machines:
        assigned_jobs_fjssp[machine].sort()
        for assigned_task in assigned_jobs_fjssp[machine]:
            start = assigned_task.start
            duration = assigned_task.duration
            dfg.append(dict(Task="AGV%s" % (machine), Start=start_date + time_delta(0, start),
                           Finish=start_date + time_delta(0, start + duration),
                           Resource="%s" % (assigned_task.job +1), complete=assigned_task.job + 1))
    colors = {}
    for i in range(num_jobs):
        key = "%s" % (i +1)
        colors[key] = "rgb(%s, %s, %s)" % (rgb(), rgb(), rgb())
    fig = ff.create_gantt(dfg, colors=colors, index_col="Resource", group_tasks=True, show_colorbar=True)
    pyplt(fig, filename=r"./GanttChart_FJSSP.html", auto_open=True)


path_of_directories = r'[..]'
cols = ['instance', 'makespan', 'solving time', 'solver']
files = []
lst = []
for names in os.listdir(path_of_directories):
    f = os.path.join(path_of_directories, names)
    files.append(f)
    if os.path.isfile(f):
        data, nb_AGVs, name = read_instance_FJSSP(f)
        instance, makespan, solving_time = flexible_jobshop(data, nb_AGVs, name)
        solver = 'ORTOOLS'
        lst.append([instance, makespan, solving_time, solver])

df1 = pd.DataFrame(lst, columns=cols)
inst_analysis = df1.to_excel('instance_analysis_ORTOOLS_FJSSP.xlsx')

print(df1)

