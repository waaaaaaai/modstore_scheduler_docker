import itertools
from util import flatten, float_geq, float_leq
from slots import Slot, ChosenSlot, ScheduledTask


class Core(object):
    def __init__(self, core_id, resource_id):
        super(Core, self).__init__()
        self.core_id = core_id
        self.resource_id = resource_id
        self.id = (resource_id, core_id)
        self.latest_time = 0
        self._schedule = [] # list for scheduling on core

    def schedule_task(self, task):
        self._schedule.append(task)
        self._schedule = sorted(self._schedule, key=lambda x: x.start) # sorted according to start time
        self.latest_time = max(self.latest_time, task.finish)

    def schedule_docker(self, docker):
        self._schedule.append(docker)
        self._schedule = sorted(self._schedule, key=lambda x: x.start) # sorted according to start time
        self.latest_time = max(self.latest_time, docker.finish)

    def unschedule_task(self, task):
        self._schedule.remove(task) #.remove
        self._schedule = sorted(self._schedule, key=lambda x: x.start)
        self.latest_time = max(t.finish for t in self._schedule) if self._schedule else 0.

    def unschedule_docker(self, docker):
        self._schedule.remove(docker) #.remove
        self._schedule = sorted(self._schedule, key=lambda x: x.start)
        self.latest_time = max(t.finish for t in self._schedule) if self._schedule else 0.

    def slots(self): ## add condition
        prev_time = 0
        for task in sorted(self._schedule, key=lambda x: x.start): #have to be sorted by task.start value
            slot_duration = task.start - prev_time
            if slot_duration > 0:
                yield Slot(start=prev_time, duration=slot_duration) #yield a slot object
            prev_time = task.finish #recursively update
        yield Slot(start=prev_time, duration=float("inf")) ## compute once

    def schedule(self):
         return sorted(self._schedule, key=lambda x: x.start)

    def schedule_fix_order(self):
        return self._schedule

    def schedule_len(self):
        return len(self._schedule)

    def get_slot(self, job, eft_r, ilft_r):    ### change to include docker factor
        """Find S(v, r): schedulable slot for v in resource r."""
        duration = job.runtime
        ret = []
        print "Slots for C{}".format(self.core_id), list(self.slots())
        for slot in self.slots():
            # spot-on: idle slot with EFT(v) <= finish time <= ILFT(v)
            #          and size >= runtime
            if slot.finish < eft_r:  # This slot is too early.
                continue
            elif slot.start < eft_r - duration:  # Truncate slot to start at EST.
                slot.start = eft_r - duration
                slot.duration = slot.finish - slot.start
            if slot.duration < duration:  # This slot is too short.
                continue
            else:  # spot on or alternative ?
                if float_leq(slot.finish, ilft_r) and float_geq(eft_r - slot.start, duration): # est_r >= slot.start
                    ret.append(ChosenSlot(finish=eft_r,
                                          duration=duration,
                                          spot_on=True, core=self, tag='type1 spot_on')) #tag is type of chosenslot
                elif float_geq(slot.finish, ilft_r) and float_geq(ilft_r - slot.start, duration):#Ilst >= slot.start
                    ret.append(ChosenSlot(finish=max(slot.start + duration, eft_r),
                                          duration=duration,
                                          spot_on=True, core=self, tag='type2 spot_on'))
                elif float_geq(slot.finish, ilft_r): # a<b or they're close
                    # otherwise, set to first available slot that can accommodate v with the finish time of the slot later than ILFT
                    ret.append(ChosenSlot(start=slot.start,
                                          duration=duration,
                                          spot_on=False, core=self, tag='alternative'))

        return ret #a list of slots given by get_slot function


class Resource(object): #define id and number of cores and combine information of each core using "chain"
    def __init__(self, resource_id, num_cores=1):
        super(Resource, self).__init__()
        self.id = resource_id
        self.cores = [Core(resource_id*num_cores+i, resource_id) for i in range(num_cores)]
    def slots(self):
        return itertools.chain(*(core.slots() for core in self.cores)) ## take all cores as iterators

    def schedule(self):
        return itertools.chain(*(core.schedule() for core in self.cores)) #TODO: incorrect:instancemethod' object is not iterable

    def get_slot(self, job, eft_r, ilft_r):
        candidate_slots = flatten(core.get_slot(job, eft_r, ilft_r) for core in self.cores) #refer to util: flatten for retrieving inner items
        docker = job.docker
        indicator=0
        for ScheduledDocker in self.schedule(): ##
            if ScheduledDocker.id == docker.id:
                indicator=1
        if indicator==0:
            for slot in candidate_slots: #TODO:rewrite ##
                if slot.start < slot.core.latest_time + docker.trans_time :
                    slot.move_start(slot.start + docker.trans_time)

        print "Candidate slots:", candidate_slots
        for slot in candidate_slots:
            if slot.spot_on:
                return slot


        return sorted(candidate_slots, key=lambda x: x.finish)[0]  #if no spot-on, return the one with earliest finish time



    def __repr__(self): #an expression of self (object)
        return "Resource(id={})".format(self.id)


