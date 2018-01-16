#!/usr/bin/env python
# Critical Path First algorithm for scheduling tasks in a workflow.
import networkx as nx
from collections import defaultdict
from slots import ScheduledTask,ChosenSlot,Slot, UploadedDocker
from resource import Resource
from util import float_geq, float_leq


class CriticalPathFirst(object):

    """CriticalPathFirst is a scheduling algorithm."""

    def __init__(self, jobs, dockers):
        super(CriticalPathFirst, self).__init__()
        self.jobs = jobs
        self.dockers = dockers  #order matters?
        self._topo_sort()
        self.find_critical_path()
        # Initialise schedule.
        self.resources = [
            Resource(resource_id=i, num_cores=2) for i in xrange(5)] #list of Resource objects
        # Resource that will be used to execute a task.
        self.assigned_resource_core = defaultdict(list)
        self.schedule_all = defaultdict(dict) ##

    def get_core(self, resource_id, core_id):
        return self.resources[resource_id].cores[core_id]

    def get_resource(self,resource_id):##
        return self.resources[resource_id]

    def cores(self):
        return [c for r in self.resources for c in r.cores]

    def _topo_sort(self):
        """
        Input: jobs.
        Output: DAG, topo sorted order.
        """
        G = nx.DiGraph()
        for job in self.jobs.itervalues():
            for parent, weight in job.parents_w.items():
                G.add_edge(parent, job.id, weight=weight)
        assert nx.is_directed_acyclic_graph(G)
        self.G = G
        self.topo_sorted_nodes = nx.topological_sort(
            G, sorted(self.jobs.keys()))
        self.topo_sorted_nodes = ['V00', 'V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V07', 'V08', 'V09', 'V10', 'V11', 'V12', 'V13']
            # correction of topological sort

    def get_trans_time(self,v,resource=None): ## TODO: rewrite
        #check nodes already scheduled on the resource(a serve) in the same docker --> return 0 or upload time

        current_docker = self.jobs[v].docker
        if current_docker is not None:

            if not self.docker_is_uploaded(v,resource):
                return float(current_docker.trans_time)
        return 0

    def docker_is_uploaded(self, v, resource): ## better take in core as arg
        if resource is not None:
            for object in resource.schedule(): #(schedule on resource /change dynamically)
                #print scheduledtask.task
                if type(object) == UploadedDocker and object.id == self.jobs[v].docker.id:
                    return True
        return False


    def find_critical_path(self):
        """
        Use DP to find the longest path.
        Input: DAG, jobs (for runtime), topo_sorted_order
        """
        max_cost = defaultdict(float)  # Time at which node finishes.
        par = {}
        for node in self.topo_sorted_nodes: #node is a str
            max_cost[node] += self.get_trans_time(node) + self.jobs[node].runtime ##
            for succ in self.G[node]:# all the children
                if max_cost[node] + self.G[node][succ]['weight'] > max_cost[succ]: #can directly use expr G[n] as in adj[n]
                    max_cost[succ] = max_cost[node] + self.G[node][succ]['weight']
                    par[succ] = node

        last_node = sorted(max_cost.keys(), key=lambda x: max_cost[x])[-1]
        self.cpl_length = max(max_cost.itervalues())
        assert self.cpl_length == max_cost[last_node]
        cpath = []
        while last_node:
            cpath.append(last_node)
            last_node = par.get(last_node, None)
        print "Critical Path Length:", self.cpl_length
        self.cpath = list(reversed(cpath))
        self.entry_node = self.cpath[0]
        self.exit_node = self.cpath[-1]
        self.cpl_parent = par

    def most_influential_parent(self, v, resource=None):
        """Predecessor that completes the communication at the latest time."""
        mip = None # initialize
        latest_finish = 0.
        for par in self.jobs[v].parents:
            finish_time = self.schedule_all[par].finish if par in self.schedule_all else self.earliest_finish(par)
            finish_time += self.get_comm_time(par, v, resource_t=resource) # btw v_j and v_i
            if latest_finish < finish_time:
                mip = par
                latest_finish = finish_time
        return mip

    def get_comm_time(self, v_i, v_j, resource_f=None, resource_t=None):
        """Returns communication cost from v_i to v_j."""
        if resource_f is None and v_i in self.schedule_all:
            resource_f = self.schedule_all[v_i].resource_id
        elif resource_f:
            resource_f = resource_f.id
        if resource_t is None and v_j in self.schedule_all:
            resource_t = self.schedule_all[v_j].resource_id
        elif resource_t:
            resource_t = resource_t.id
        if resource_f is not None and resource_t is not None and resource_f == resource_t:
            return 0. #meaning that v_j=v_i
        return self.jobs[v_i].children_w[v_j] # children weight = comm_time

    def earliest_start(self,id,resource=None,core=None):
        if id in self.jobs:
            return self.earliest_start_task(id,resource,core)
        else:
            return self.earliest_start_docker(id,resource,core)


    def earliest_start_task(self, v, resource=None,core=None): # EST depends on each core
        """EST - earliest start time."""
        if v == self.entry_node:
            return 0.
        # Calculate most influential parent.
        mip = self.most_influential_parent(v, resource=resource) ##
        #if not self.docker_is_uploaded(v,core=resource.cores[0]) and self.jobs[v].docker.trans_time+ >self.earliest_finish(mip) + self.get_comm_time(mip, v, resource_t=resource):
         #return self.jobs[v].docker.trans_time +
        #else:
        return self.earliest_finish(mip) + self.get_comm_time(mip, v, resource_t=resource)

    def earliest_start_docker(self, d, resource=None,core=None): # EST depends on each core

        return 0

    def earliest_finish(self,id,resource=None,curr_resource=None):
        if id in self.jobs:
            return self.earliest_finish_task(id,resource)
        else:
            return self.earliest_finish_docker(id,resource,curr_resource)

    def earliest_finish_task(self, v, resource=None):
        """
        EFT - earliest finish time.
        EFT(v) = EST(v) + compute_cost(v).
        """
        # TODO: see if this is legit.
        if v in self.schedule_all:
            return self.schedule_all[v].finish
        return self.earliest_start(v, resource=resource) + self.jobs[v].runtime

    def earliest_finish_docker(self, d, resource=None,curr_resource=None):
        """
        EFT - earliest finish time.
        EFT(v) = EST(v) + compute_cost(v).
        """
        # TODO: see if this is legit.
        if curr_resource is not None:
            return self.schedule_all[d][curr_resource].finish #when it's AFT
        return self.earliest_start_docker(d, resource=resource) + self.dockers[d].trans_time

    def get_partner_tasks(self, v): # partner
        partner_tasks = set()
        for child in self.jobs[v].children:
            for parent in self.jobs[child].parents:
                if parent != v:
                    partner_tasks.add(parent)
        return partner_tasks

    def max_comm_time(self, v): #

        if self.jobs[v].children_w.values():
            return max(self.jobs[v].children_w.values())
        else:
            return 0
#-----
    def actual_finish(self, id, resource,curr_resource=None,recompute=False):
        if id in self.jobs:
            return self.actual_finish_task(id,resource,recompute)
        else:
            return self.actual_finish_docker(id,resource,curr_resource,recompute)

    def actual_finish_task(self, v, resource, recompute=False):
        # TODO: check logic.
        if v in self.schedule_all and not recompute:
            return self.schedule_all[v].finish
        aft = 0.
        for p, weight in self.jobs[v].parents_w.items():
            if p in self.schedule_all:
                task = self.schedule_all[p]

                aft = max(aft, task.finish + (self.get_comm_time(p, v) # aft is max finish time
                                              # of all parents , it's a loop to find max
                                              if task.resource_id != resource.id else 0.))
        return aft + self.jobs[v].runtime

    def actual_finish_docker(self, d, resource, curr_resource,recompute=False):
        # TODO: check logic.
            return self.schedule_all[d][curr_resource].finish
#-----

    def actual_latest_completion_time(self, v, resource, recompute=False):
        return (self.actual_finish(id=v, resource=resource, recompute=recompute) +
                self.max_comm_time(v)) # latest completion = AFT + max_comm   #(5)

    def latest_completion_time(self, v):
        if v in self.schedule_all:
            return self.schedule_all[v].finish
        return self.earliest_finish(v) + self.max_comm_time(v) # if not scheduled, latest completion = EFT + max_comm


#-----
    def interim_latest_finish(self,id,resource,verbose=False):
        if id in self.jobs:
            return self.interim_latest_finish_task(id,resource,verbose)
        else:
            return self.interim_latest_finish_docker(id,resource,verbose)

    def interim_latest_finish_task(self, v, resource, verbose=False):
        recompute = False
        max_r = self.actual_latest_completion_time(v, resource, recompute=recompute)
        partner_tasks = self.get_partner_tasks(v)
        if partner_tasks:
            max_r = max(max_r,
                        max(self.latest_completion_time(p) for p in partner_tasks)) #(7)
        if verbose:
            print "        ILFT({v}, {resource}) = AFT ({aft}) + max(LCT [{lct}], max(LCT_partner {lct_p}) - LCT {lct})".format(
                v=v,
                resource=resource.id,
                aft=self.actual_finish(v, resource, recompute=recompute),
                lct=self.actual_latest_completion_time(v, resource),
                lct_p=max(self.latest_completion_time(
                    partner) for partner in self.get_partner_tasks(v)) if partner_tasks else "[none]",
            )
        return (self.actual_finish(v, resource, recompute=recompute)
                + max_r
                - self.actual_latest_completion_time(v, resource, recompute=recompute))

    def interim_latest_finish_docker(self, d, resource, verbose=False):
        lst=[]
        for task in self.dockers[d].tasks:
            lst.append(self.interim_latest_finish(task,resource,verbose))
        return min(lst)
#------
    def actual_latest_start(self,id,resource,verbose=False):
        if id in self.jobs:
            return self.actual_latest_start_task(id,resource,verbose)
        else:
            return self.actual_lastest_start_docker(id,resource,verbose)

    def actual_latest_start_task(self, v, resource, verbose=False):
        return self.interim_latest_finish(v, resource, verbose) - self.jobs[v].runtime

    def actual_lastest_start_docker(self,d,resource,verbose=False):
        return self.interim_latest_finish_docker(d, resource, verbose) - self.dockers[d].trans_time
#-----
    def makespan_increase(self, v, slot, modify=False):  #include function to print updated schedule
        """recalculate finish times of CP child tasks and CP tasks after these CP child tasks."""
        cp_child_tasks = self.jobs[v].children & self.cpath_tasks # intersection
        after_pt = False
        last_finish = 0.
        mi = 0.
        for cp_task in self.get_core(0, 0).schedule():# cp_task as scheduledTask Class: 3->7
            if cp_task.id in cp_child_tasks:
                # new earliest start time.
                est = slot.finish + self.get_comm_time(v, cp_task.id) # slot as an input
                mi = max(mi, est - cp_task.start) #cp_task.start= original start time
                if modify and cp_task.start < est:
                    cp_task.move_start(est) #move start time to EST
                    print "setA {}".format(cp_task)
            elif last_finish > cp_task.start and modify:
                cp_task.move_start(last_finish)
                print "setB {}".format(cp_task)
            last_finish = cp_task.finish
        if modify:
            self.get_core(0, 0).latest_finish = last_finish #TODO: check functionality
        return mi

    def schedule_task(self, v, core, slot_start):
        self.assigned_resource_core[v] = core.id
        task_scheduled = ScheduledTask(task=self.jobs[v],
                                       start=slot_start,
                                       core=core)


        self.schedule_all[v] = task_scheduled
        core.schedule_task(task_scheduled)

        return task_scheduled

    def schedule_docker(self, d, core, slot_start):
        self.assigned_resource_core[d].append(core.id)
        docker_uploaded = UploadedDocker(docker=self.dockers[d],
                                       start=slot_start,
                                       core=core)


        self.schedule_all[d][docker_uploaded.resource_id] = (docker_uploaded) #put in a dict with keys = resource.id
        core.schedule_docker(docker_uploaded)

        return docker_uploaded


    def reschedule_task(self, task):
        v = task.id
        old_core = self.schedule_all[v].core
        old_core.unschedule_task(self.schedule_all[v])
        v = task.id
        self.assigned_resource_core[v] = task.core.id
        self.schedule_all[v] = task
        task.core.schedule_task(task)

    def reschedule_docker(self, new_docker, curr_docker): ###
        new_is_better = True
        d = new_docker.id
        new_resource = new_docker.resource_id
        old_resource = curr_docker.resource_id

        for object in self.resources[new_docker.resource_id].schedule(): #Remove multiple docker
            if object.id == new_docker.id and (object.finish!=new_docker.finish or object.core_id!=new_docker.core_id):
                if object.finish > new_docker.finish:

                    print ">>Remove multiple docker {}".format(object)
                    object.core.unschedule_docker(object)
                    del self.schedule_all[object.id][object.resource_id]

                else:
                    new_is_better = False

        if new_is_better:
            preserve=False  #whether to preserve the old_docker on original place
            old_core = curr_docker.core
            for item in self.resources[curr_docker.resource_id].schedule(): ##check whether to move or to make new copy
                if type(item) == ScheduledTask and item.docker.id == curr_docker.id:
                    preserve = True

            self.assigned_resource_core[d].append(new_docker.core.id)
            self.schedule_all[d][new_resource] = new_docker
            new_core = new_docker.core
            new_core.schedule_docker(new_docker)
            if preserve == False:
                old_core.unschedule_docker(curr_docker)
                del self.schedule_all[d][old_resource]


    def pushdown(self, core):
        """push down subsequent tasks on the same core"""
        print "Pushing down tasks on core", core.id
        last_finish = 0.
        for curr_obj in core.schedule_fix_order():  ##important!!!
            if last_finish > curr_obj.start:
                print "> Pushing down {} from {:g} to {:g}".format(curr_obj.id, curr_obj.start, last_finish)
                curr_obj.move_start(last_finish) #choosenslot obj
                print core.schedule()

                if type(curr_obj) == ScheduledTask: ####
                    for child in self.jobs[curr_obj.id].children:
                        if child in self.schedule_all:
                            child_task = self.schedule_all[child]
                            earliest_start = curr_obj.finish + self.get_comm_time(curr_obj.id, child) #pushdown child as well
                            #print "> Parent{} Child{}".format(curr_obj.id, child_task.id)
                            if earliest_start > child_task.start:
                                child_task.move_start(earliest_start)
                                self.pushdown(child_task.core)
                        # next_cores.add(self.schedule_all[child].core)
                elif type(curr_obj) == UploadedDocker:
                    for child in self.dockers[curr_obj.id].tasks:
                        if child in self.schedule_all and curr_obj.resource_id == self.schedule_all[child].resource_id: #only task on same resource will be affected
                            child_task = self.schedule_all[child]
                            earliest_start = curr_obj.finish
                            #print "> Docker{} Content{}".format(curr_obj.id, child_task.id)
                            if earliest_start > child_task.start:
                                child_task.move_start(earliest_start)
                                self.pushdown(child_task.core)
            last_finish = curr_obj.finish
        core.latest_finish = last_finish
        # for c in next_cores:
        #     self.pushdown(c)



    def compute(self):  ###
        # Put all tasks in the critical path in a dedicated resource. ## follows algorithm 1
        for cp_task in self.cpath: # assign CP tasks to rCP // e.g., r0,0



            if not self.docker_is_uploaded(cp_task,self.get_resource(0)): ##same as self.get_core(0) ##order matters
                docker_uploaded = self.schedule_docker(self.jobs[cp_task].docker.id, self.get_core(0,0),self.get_core(0,0).latest_time)
                print "Uploading Docker{}".format(docker_uploaded)

            task_scheduled = self.schedule_task(
                cp_task, self.get_core(0, 0), self.get_core(0, 0).latest_time )

            print "Scheduling CP Task {}".format(task_scheduled)
        self.cpath_tasks = set(self.cpath)

        G_minus_cp_tasks = set(
            self.G.subgraph(v for v in self.G.nodes() if v not in self.cpath_tasks))
        # Process tasks in topological order
        G_minus_cp_tasks = filter(
            lambda v: v in G_minus_cp_tasks, self.topo_sorted_nodes) # filter
        assert len(G_minus_cp_tasks) + len(self.cpath) == len(self.G)

        print "Critical Path:", ", ".join(self.cpath)
        print "------------------------------------------------------------"

        # This is an empty resource.
        resource_curr = 1  # r^new = r^{CP+1}

        # R' - candidate resources. Iteratively increase this.
        candidate_resources = self.resources[
            :resource_curr + 1]  # R' = {r^CP, r^new}
        candidate_cores = [c for r in candidate_resources for c in r.cores]

        for v in G_minus_cp_tasks:
            # r*: best spot-on resource.
            r_star = None
            least_makespan_increase = None

            # r': best alternative resource.
            r_prime = None

            curr_job = self.jobs[v]
            curr_duration = curr_job.runtime

            # Compute EFT(v) and ILFT(v) on (coarse-grained) resources in R'.
            # Use assigned_resource for actual time/resource_core_id.
            est = self.earliest_start(v)
            eft = self.earliest_finish(v)
            print "Scheduling Task {} (EST: {}, EFT: {}, Duration: {}, MIP: {}, Belongs to Docker: {} )".format(v, est, eft, curr_duration, self.most_influential_parent(v),curr_job.docker.id)
            est, eft = None, None
            efts = {r.id: self.earliest_finish(v, r)
                    for r in candidate_resources}
            for r_id, l_eft in efts.items():
                print "        EFT({}, {}) = {}".format(v, r_id, l_eft)
            ilft = {r.id: self.interim_latest_finish(id=v, resource=r, verbose=True) for r in candidate_resources}
            print "        Partner tasks:", list(self.get_partner_tasks(v))
            print "        Parent tasks:", curr_job.parents
            print "        Child tasks:", curr_job.children

            cp_child_tasks = curr_job.children & self.cpath_tasks
            print "        CP child tasks:", cp_child_tasks

            for resource in candidate_resources:

                print "==Resource {}==".format(resource.id)
                chosen_slot = resource.get_slot(          # find S(v,r) // schedulable slot for v
                    curr_job, eft_r=efts[resource.id], ilft_r=ilft[resource.id])
                assert chosen_slot is not None
                # chosen_slot_finish = AFT(v, r_i)
                print "        Chosen slot: ({})".format(chosen_slot.nice())

                # children(v) \intersect CP == null and S(v,r) is spot-on
                if len(cp_child_tasks) == 0 and chosen_slot.spot_on:
                    # Task without any direct precedence constraint with CP
                    # tasks (tasks without CP child tasks)
                    #
                    # First-fit: Assign task to first resource that completes
                    # execution before the ILFT of that task. ... unless the
                    # best spot-on resource is a new resource.
                    #
                    # Rationale: no immediate impact on makespan at time of
                    # scheduling(confirmed) - focus on condensing the schedule to reduce
                    # resource usage.
                    print "No CP child tasks"
                    # TODO: determine if correct.
                    r_star = chosen_slot
                    break
                elif chosen_slot.spot_on:  ## with CP child
                    # Schedule in best-fit fashion based on makespan increase.
                    # Assign task to the resource on which its finish time is no
                    # later than its ILFT and the makespan of the current
                    # partial schedule is best preserved (no or minimum
                    # increase).

                    # Compute MI(v, r): Makespan increase.
                    makespan_increase = self.makespan_increase(v, chosen_slot)
                    # update r* based on makespan increase
                    if r_star is None or least_makespan_increase > makespan_increase:    #initial value
                        print "Update r* to {}: (best-fit, makespan increase: {})".format(chosen_slot, makespan_increase)
                        least_makespan_increase = makespan_increase
                        r_star = chosen_slot
                else:
                    # update r' based on AFT.
                    if r_prime is None or chosen_slot.finish < r_prime.finish:
                        print "Update r' to {}".format(chosen_slot)
                        r_prime = chosen_slot

            print "Finishes:", r_star.finish if r_star else "none", r_prime.finish if r_prime else "none"
            if r_star.resource_id == resource_curr:
                if (
                   r_prime is not None and float_geq(r_star.finish, r_prime.finish) and
                   (len(cp_child_tasks) == 0 or
                    (len(cp_child_tasks) > 0
                     and float_geq(self.makespan_increase(v, r_star), self.makespan_increase(v, r_prime))))
                   ):
                    # Compare with best alternative resource.
                    #assert float_geq(r_star.finish, ilft[r_star.resource_id])
                    #assert float_leq(r_star.finish, r_prime.finish)
                    r_star = r_prime
                    print "Compared with best alternative resource (r'), using it"
                else:
                    resource_curr += 1
                    candidate_resources = self.resources[:resource_curr + 1]
                    candidate_cores = [
                        c for r in candidate_resources for c in r.cores]
                    print "Using new resource"

            assert r_star is not None
            print "r*", r_star.core_id, "(Finish: {})".format(r_star.finish)
            assert r_star.finish is not None
            assert float_geq(r_star.finish, eft)
            if r_prime:
                print "r'", r_prime.core_id, "(Finish: {})".format(r_prime.finish)
            else:
                print "r': none"

            previous_latest = r_star.core.latest_time #latest time on core before uploading current task
            task = self.schedule_task(v, r_star.core, r_star.start)   ## schedule task !

            #upload docker
            if not self.docker_is_uploaded(v,self.get_resource(r_star.resource_id)):
                print "Uploading Docker {}".format(self.jobs[v].docker.id)
                docker_uploaded = self.schedule_docker(self.jobs[v].docker.id,r_star.core,previous_latest)
                #instantly reschedule slot for docker
                r_temp, v_temp, docker_rescheduled = self.find_inefficiency_slot_docker(docker_uploaded,task)
                if docker_rescheduled is not None:
                    self.reschedule_docker(docker_rescheduled,docker_uploaded)
                    self.pushdown(docker_rescheduled.core)
                #r_star.move_start(r_star.core.latest_time)

            print "FINAL: Scheduled Task {} on {}".format(task, task.core_id)

            # Update schedule - loosen.
            self.makespan_increase(v, r_star, modify=True)
            self.check_assumptions(candidate_cores)
            print "----------------------------"

        #assert len(self.schedule_all) == len(self.jobs) + len(self.dockers)
        self.compact()
        self.check_assumptions()
        return self.schedule_all

    def check_and_print_schedule(self, candidate_cores):
        for core in candidate_cores:

            print core.id, ", ".join("[{}-{}: {}]".format(d.start, d.finish,d.id ) for d in core.schedule())
            last_finish = 0
            for d in core.schedule():
               # assert float_geq(d.start, last_finish)
                last_finish = d.finish

    def check_assumptions(self, candidate_cores=None):
        if candidate_cores is None:
            candidate_cores = self.cores()

        self.check_and_print_schedule(candidate_cores)

        print

        for job_id, job_i in self.jobs.items():
            for parent in job_i.parents:
                if parent not in self.schedule_all or job_id not in self.schedule_all:
                    continue
                # print parent, job_id, self.schedule_all[parent], self.schedule_all[job_id]
                #print parent, job_id, self.schedule_all[parent].finish, self.get_comm_time(parent, job_id), self.schedule_all[job_id].start
                #assert float_leq(self.schedule_all[parent].finish, self.schedule_all[job_id].start)
                #assert float_leq(self.schedule_all[parent].finish + self.get_comm_time(parent, job_id), self.schedule_all[job_id].start)

#-------
    def latest_start(self,id,resource=None):
        if id in self.jobs:
            return self.latest_start_task(id,resource)
        else:
            return self.latest_start_docker(id,resource)

    def latest_start_task(self, v,resource=None):
        if v in self.jobs:
            return self.latest_finish(v,resource) - self.jobs[v].runtime

    def latest_start_docker(self, d,resource=None):
        if d in self.dockers:
            return self.latest_finish_docker(d,resource) - self.dockers[d].trans_time

#-------

    def latest_finish(self,id,resource=None):
        if id in self.jobs:
            return self.latest_finish_task(id,resource)
        else:
            return self.latest_finish_docker(id,resource)

    def latest_finish_task(self, v, resource=None):
        if v == self.exit_node:
            return self.earliest_finish(v, resource)
        return min(self.latest_start(v_k) - self.get_comm_time(v, v_k, resource_f=resource)
                   for v_k in self.jobs[v].children)
    def latest_finish_docker(self,d,resource=None):
        lst=[]
        for task in self.dockers[d].tasks:
            lst.append(self.latest_start(task))
        return min(lst)
#-------

    # def insert(self,r_star,v_star,new,):
    #     if r_star is not None and v_star is not None: # TODO
    #         # move v_i after v* on r*
    #         print "move v_i ({}) after v* ({}) on r* ({})".format(object.id, v_star, r_star)
    #         self.reschedule_task(new)
    #         self.pushdown(new.core)
    #         # Push down tasks - affected: child, parent tasks of v_i, subsqeuently scheduled tasks after v*
    #     elif r_star is not None:
    #         # move v_i at the beginning of r*
    #         print "move v_i at the beginning of r*", v_star, r_star
    #         self.reschedule_task(new)
    #         self.pushdown(new.core)
    #     else:
    #         print "breaking - no more"
    #         break



    def need_reschedule_docker(self,task):
        if task is not None:
            curr_resource = task.resource_id
            for object in self.resources[curr_resource].schedule():
                if object.id == task.docker.id and float_leq(object.finish, task.start):
                    return False

        return True

    def remove_multiple(self,docker):
        for object in self.resources[docker.resource_id].schedule():
            if object.id == docker.id and (object.finish!=docker.finish or object.core_id!=docker.core_id):
                if object.finish > docker.finish:

                    print ">>Remove multiple docker {}".format(object)
                    object.core.unschedule_docker(object)
                    del self.schedule_all[object.id][object.resource_id][object]
                else:
                    print ">>Remove multiple docker {}".format(docker)
                    docker.core.unschedule_docker(docker)
                    del self.schedule_all[docker.id][docker.resource_id][docker]

    def compact(self):
        for r in reversed(self.resources):
            # compute LSTs/LFTs and ALSTs/ALFTs
            for core in reversed(r.cores):
                for object in reversed(core.schedule()):##
                    if type(object) == ScheduledTask: ##
                        r_star, v_star, new_task = self.find_inefficiency_slot(object)
                        found = True
                        if r_star is not None and v_star is not None: # TODO
                            # move v_i after v* on r*
                            print "move v_i ({}) after v* ({}) on r* ({})".format(new_task.id, v_star, r_star)
                            self.reschedule_task(new_task)
                            self.pushdown(new_task.core)
                            # Push down tasks - affected: child, parent tasks of v_i, subsqeuently scheduled tasks after v*
                        elif r_star is not None:
                            # move v_i at the beginning of r*
                            print "move v_i at the beginning of r*", v_star, r_star
                            self.reschedule_task(new_task)
                            self.pushdown(new_task.core)
                        else:
                            print "breaking - no more"
                            found = False #whether to write return here
                        self.check_assumptions()

                        if found:
                            for core_2 in reversed(r.cores):
                                for item in reversed(core_2.schedule()):##
                                    if type(item) == UploadedDocker and item.id == object.docker.id:# UploadedDocker
                                        if self.need_reschedule_docker(new_task):
                                            r_prime, v_prime, new_docker = self.find_inefficiency_slot(item,new_task) #
                                            if r_prime is not None and v_prime is not None:
                                                # move v_i after v* on r*
                                                print "move Docker ({}) after v_prime ({}) on r_prime ({})".format(new_docker.id, v_prime, r_prime)
                                                self.reschedule_docker(new_docker,item)
                                                self.pushdown(new_docker.core)
                                            elif r_prime is not None: #most cases
                                                # move v_i at the beginning of r*
                                                print "move d_i at the beginning of r'", v_prime, r_prime
                                                self.reschedule_docker(new_docker,item)
                                                self.pushdown(new_docker.core)
                                            else:
                                                print "breaking - no more"
                                                    ## Reversely find children tasks in docker
                                            self.check_assumptions()
                                        else:
                                            stay = False
                                            for others in self.resources[item.resource_id].schedule():
                                                if type(others)==ScheduledTask and others.docker.id == item.id:
                                                    stay = True
                                            if stay == False:
                                                print
                                                print "Docker {} can be deleted".format(item)
                                                item.core.unschedule_docker(item)
                                                print
                                                self.check_assumptions()

                        # if no slot?
                # update info of affected tasks
                    print
        # merge partly used resources


#-----
    def find_inefficiency_slot(self,object,extra=None):

        if type(object)==ScheduledTask:
            return self.find_inefficiency_slot_task(object)
        else:
            return self.find_inefficiency_slot_docker(object,extra)


    def find_inefficiency_slot_task(self, task_i): #unique
        v = task_i.id
        r_ret = None
        v_ret = None
        chosen_slot = None
        print "Finding inefficiency slot for task", task_i
        for resource in self.resources:
            if r_ret is not None:
                break
            # TODO: figure if this should be at core/resource level.
            if  resource.id == self.schedule_all[v].resource_id:
                continue   #must not skip self resource in this case

            print
            # alst_i = self.actual_latest_start(v, resource, verbose=True)
            # TODO: see if ALST should be taken from LST or ILST
            alst_i = self.latest_start(v,resource)  #TODO: wrong
            print "ALST({}, {}) = {}".format(v, resource.id, alst_i)

            if all(len(core.schedule()) == 0 for core in resource.cores): #all function
                continue

            for core in resource.cores: ##better than searching from beginning?
                if len(core.schedule()) == 0:
                    continue
                print "> -- Checking core --", core.id
                schedule = core.schedule()

                # Check for explicit slot (schedulable slot at beginning of
                # resource schedule)
                # EST(r) >= EFT(v_i, r):
                first_slot = core.slots().next()
                if first_slot.start == 0. and float_geq(first_slot.finish, self.earliest_finish(v)):
                    print "explicit slot found in", resource.id, core.id
                    print list(core.slots())
                    print self.earliest_finish(v, resource)
                    r_ret = resource
                    chosen_slot = ScheduledTask(task=self.jobs[v],
                                                finish=self.earliest_finish(v, resource),
                                                core=core)
                    break

                for i, cur_task in enumerate(schedule):
                    v_ii = i + 1
                    next_task = schedule[v_ii] if v_ii < len(schedule) else None
                    print ">> Trying insertion b/w {}th and {}th scheduled object ({}, {})".format(i, v_ii, cur_task, next_task)

                    # compute AST(v_i, r) after AFT(v',r)
                    ast_i = self.actual_finish(
                        task_i.id, resource, recompute=True) - task_i.duration
                    print ">>>", "AST(v_i, r):", ast_i, cur_task.finish
                    ast_i = max(ast_i, cur_task.finish)
                    assert float_geq(ast_i, cur_task.finish)
                    aft_i = ast_i + task_i.duration

                    if next_task is not None:
                        lst_ii = self.latest_start(next_task.id)
                        print "LST(v'', r) = {}".format(lst_ii)

                    # AST (v_i, r) >= ALST (v_i, r)
                    if float_geq(ast_i, alst_i):
                        break
                    # AST(next_task, r) - AST(task, r) >= w_i
                    elif next_task is None or float_geq(next_task.start - ast_i, task_i.duration):
                        # explicit slot found
                        r_ret = resource
                        v_ret = cur_task.id
                        chosen_slot = ScheduledTask(task=self.jobs[v],
                                                    start=ast_i,
                                                    core=core)
                        print "QWEQWEQ", cur_task, next_task
                        print "explicit slot found", core.id, v_ret

                        print "Slot chosen:", chosen_slot
                        return r_ret, v_ret, chosen_slot
                    # [AD] LST(next_task, r) - AST(task, r) >= w_i
                    elif float_geq(lst_ii - ast_i, task_i.duration):
                        # v'' (next_task) can be pushed down to accommodate v_i
                        # however, actual feasibility of v_i before v'' may not be guaranteed due to subsequently scheduled tasks
                        # recursivley check subsequent tasks to see if they can pushed down
                        # to eventually make a slot sufficient for v_i
                        t_push = aft_i - self.actual_latest_start(next_task.id, resource)  # AFT(task, r) - ALST(next_task, r)
                        print ">>> Checking if tasks can be pushed down: t_push {} = AST(v_i, r) ({}) - ALST(v'', r) ({})".format(t_push, aft_i, self.actual_latest_start(next_task.id, resource))
                        while t_push > 0:
                            v_ii = v_ii + 1
                            next_task = schedule[v_ii] if v_ii < len(schedule) else None
                            print ">>>> Checking subsequent task {}".format(next_task)
                            if next_task is not None:
                                print ">>>> LST(v'' [{}], r) = {}, AST(v'', r) = {}".format(v_ii, self.latest_start(next_task.id), next_task.start)
                            else:
                                print "next_task: None"
                            if next_task is None:  # no more subsequent task
                                t_push = 0
                            # [AD] LST(next_task, r) - AST(next_task, r) >= t_push
                            elif float_geq(self.latest_start(next_task.id) - next_task.start, t_push):
                                # inefficiency slot is realized.
                                # ALST(v'', r) - AST(v'', r)
                                t_push -= self.actual_latest_start(next_task.id, resource) - next_task.start  # ALST(next_task, r) - AST(next_task, r)
                            else:
                                # subsequent task cannot be pushed down enough
                                # for remaining amount of time (t_push)
                                break
                        if float_leq(t_push, 0):
                            # implicit slot found.
                            r_ret = resource
                            v_ret = cur_task
                            chosen_slot = ScheduledTask(task=self.jobs[v],
                                                        start=ast_i,
                                                        core=core)
                            print "implicit slot, t_push: {}".format(t_push), cur_task
                            print "Slot chosen:", chosen_slot
                            return r_ret, v_ret, chosen_slot

                print
        print "Slot chosen:", chosen_slot
        return r_ret, v_ret, chosen_slot


    def find_inefficiency_slot_docker(self, docker_i,task_i): #unique
        d = docker_i.id
        curr_resource= docker_i.resource_id


        r_ret = None
        v_ret = None
        chosen_slot = None

        print "Finding inefficiency slot for docker", docker_i
        if task_i is not None:
            resource = self.resources[task_i.resource_id]
        else:
            resource = self.resources[docker_i.resource_id]
        print
        # alst_i = self.actual_latest_start(v, resource, verbose=True)
        # TODO: see if ALST should be taken from LST or ILST
        alst_i = self.latest_start(d)
        print "ALST({}, {}) = {}".format(d, resource.id, alst_i) #40



        for core in resource.cores:
            if len(core.schedule()) == 0:
                continue
            print "> -- Checking core --", core.id
            schedule = core.schedule()

            #
            #
            #
            #  for explicit slot (schedulable slot at beginning of
            # resource schedule)
            # EST(r) >= EFT(v_i, r):
            first_slot = core.slots().next()
            print "First slot:", first_slot
            if first_slot.start == 0. and float_geq(first_slot.finish, self.earliest_finish(d)):#TODO
                print "explicit slot found in", resource.id, core.id
                print list(core.slots())
                print self.earliest_finish(d, resource)
                r_ret = resource
                chosen_slot = UploadedDocker(docker=self.dockers[d],
                                            finish=self.earliest_finish(d, resource),
                                            core=core)
                break

            for i, cur_task in enumerate(schedule):
                v_ii = i + 1
                next_obj = schedule[v_ii] if v_ii < len(schedule) else None
                print ">> Trying insertion b/w {}th and {}th object ({}, {})".format(i, v_ii, cur_task, next_obj)

                # compute AST(v_i, r) after AFT(v',r)
                ast_i = self.actual_finish(
                    docker_i.id, resource,curr_resource, recompute=True) - docker_i.duration
                print ">>>", "AST(curr_docker, r):", ast_i, cur_task.finish
                ast_i = cur_task.finish#7 # ????
                #assert float_geq(ast_i, cur_task.finish)
                aft_i = ast_i + docker_i.duration #11


                if next_obj is not None:
                    lst_ii = self.latest_start(next_obj.id)
                    print "LST(next_object:{}, r) = {}".format(next_obj.id,lst_ii)


                # AST (v_i, r) >= ALST (v_i, r)
                if float_geq(ast_i, alst_i):
                    break
                # AST(next_obj, r) - AST(task, r) >= w_i
                elif next_obj is None or float_geq(next_obj.start, docker_i.duration + ast_i):
                    # explicit slot found
                    r_ret = resource
                    v_ret = cur_task.id
                    chosen_slot = UploadedDocker(docker=self.dockers[d],
                                                start=ast_i,
                                                core=core)
                    print "QWEQWEQ", cur_task, next_obj
                    print "explicit slot found", core.id, v_ret
                    break
                # [AD] LST(next_obj, r) - AST(task, r) >= w_i
                elif float_geq(lst_ii - ast_i, docker_i.duration):
                    # v'' (next_obj) can be pushed down to accommodate v_i
                    # however, actual feasibility of v_i before v'' may not be guaranteed due to subsequently scheduled tasks
                    # recursivley check subsequent tasks to see if they can pushed down
                    # to eventually make a slot sufficient for v_i
                    t_push = aft_i - self.actual_latest_start(next_obj.id, resource)  # AFT(task, r) - ALST(next_obj, r)
                    print ">>> Checking if tasks can be pushed down: t_push {} = AST(v_i, r) ({}) - ALST(v'', r) ({})".format(t_push, aft_i, self.actual_latest_start(next_obj.id, resource))
                    while t_push > 0: #need to check subsequent task, otherwise enough space
                        v_ii = v_ii + 1
                        next_obj = schedule[v_ii] if v_ii < len(schedule) else None
                        print ">>>> Checking subsequent task {}".format(next_obj)
                        if next_obj is not None:
                            print ">>>> LST(v'' [{}], r) = {}, AST(v'', r) = {}".format(v_ii, self.latest_start(next_obj.id), next_obj.start)
                        else:
                            print "next_obj: None"
                        if next_obj is None:  # no more subsequent task
                            t_push = 0
                        # [AD] LST(next_obj, r) - AST(next_obj, r) >= t_push
                        elif float_geq(self.latest_start(next_obj.id) - next_obj.start, t_push):
                            # inefficiency slot is realized.
                            # ALST(v'', r) - AST(v'', r)
                            t_push -= self.actual_latest_start(next_obj.id, resource) - next_obj.start  # ALST(next_obj, r) - AST(next_obj, r)
                        else:
                            # subsequent task cannot be pushed down enough
                            # for remaining amount of time (t_push)
                            break
                    if float_leq(t_push, 0): #t_push < 0
                        # implicit slot found.
                        r_ret = resource
                        v_ret = cur_task
                        chosen_slot = UploadedDocker(docker=self.dockers[d],
                                                    start=ast_i,
                                                    core=core)
                        print "implicit slot realised, t_push: {}".format(t_push), cur_task
                        break
                print
        print "Slot chosen:", chosen_slot
        return r_ret, v_ret, chosen_slot
