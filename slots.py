
class Slot(object): # create slot object
    def __init__(self, start=None, finish=None, duration=None, tag=''):
        super(Slot, self).__init__()
        self.tag = tag
        if finish is None and duration is not None:
            self.finish = start + duration
            self.duration = duration
            self.start = start
        elif finish is not None and duration is None:
            self.finish = finish
            self.duration = finish - start
            self.start = start
        elif start is None and finish and duration:
            self.start = finish - duration
            self.duration = duration
            self.finish = finish
        #else:
           # assert False

    def move_start(self, new_start): #to new start
        self.start = new_start
        self.finish = new_start + self.duration #make sense

    def __repr__(self):
        return "{{{:g}->{:g}{}}}".format(self.start, self.finish, self.tag)


class ChosenSlot(Slot):
    def __init__(self, spot_on=None, core=None, *args, **kwargs):
        super(ChosenSlot, self).__init__(*args, **kwargs)
        self.spot_on = spot_on
        self.core_id = core.id
        self.resource_id = core.resource_id
        self.core = core
        self.task = None
    def __repr__(self):
        return super(ChosenSlot, self).__repr__() + '[' + ('T' if self.spot_on else 'F') + '][C{}]'.format(self.core_id)
        # return "{{{:g}->{:g}, {}}}".format(self.start, self.finish, 'T' if self.spot_on else 'F')

    def nice(self):
        return self.__repr__() #?


class ScheduledTask(ChosenSlot):
    def __init__(self, task, core, *args, **kwargs): #3rd tier include args task and core from ChosenSlot class
        kwargs['duration'] = task.runtime
        kwargs['core'] = core  #core as a kwarg
        super(ScheduledTask, self).__init__(*args, **kwargs) #call upper class for predefined variables
        self.id = task.id
        self.core = core
        self.core_id = core.id
        self.resource_id = core.resource_id
        self.docker = task.docker


    def __repr__(self):
        return "{{{}: {:g}->{:g}, {} on C{}}}".format(self.id, self.start, self.finish, 'T' if self.spot_on else 'F', self.core_id)

##
class UploadedDocker(ChosenSlot):
    def __init__(self, docker, core, *args, **kwargs): #3rd tier include args task and core from ChosenSlot class
        kwargs['duration'] = docker.trans_time
        kwargs['core'] = core  #core as a kwarg
        super(UploadedDocker, self).__init__(*args, **kwargs) #call upper class for predefined variables
        self.id = docker.id
        self.core = core
        self.core_id = core.id
        self.resource_id = core.resource_id
        self.tasks=docker.tasks

    def __repr__(self):
        return "{{{}: {:g}->{:g}, {} on C{}}}".format(self.id, self.start, self.finish, 'T' if self.spot_on else 'F', self.core_id)