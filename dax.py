#!/usr/bin/env python
# Reader for DAX format
import xml.etree.cElementTree as ET
from collections import defaultdict


class Job(object):
    def __init__(self, attrib, uses, docker=None):
        super(Job, self).__init__()
        self.uses = uses
        if 'id' in attrib:
            self.id = attrib['id']
        if 'name' in attrib:
            self.name = attrib['name']
        self.runtime = float(attrib['runtime'])
        self.attrib = attrib
        self.docker = docker
        self.parents = set()
        self.parents_w = {}
        self.children = set()
        self.children_w = {}

    def add_parent(self, parent, weight):
        self.parents.add(parent)
        self.parents_w[parent] = float(weight)

    def add_child(self, child, weight):
        self.children.add(child)
        self.children_w[child] = float(weight)

    def __str__(self):
        return 'Job{id: {id}}'.format(id=self.attrib['id'])


def read_file(filename):
    NS = '{http://pegasus.isi.edu/schema/DAX}'
    tree = ET.parse(filename)
    root = tree.getroot()
    jobs = {}
    parents = defaultdict(set)
    for job in root.iter(NS+'job'):
        uses = []
        for use in job.iter(NS+'uses'):
            uses.append(use.attrib) #attrib = a dict
        jobs[job.attrib['id']] = Job(job.attrib, uses)

    for child in root.iter(NS+'child'):
        input_sizes = {}
        for input_file in child.iter(NS + 'uses'):
            if input_file.attrib['link'] == 'input':
                input_sizes[input_file.attrib['file']] = input_file.attrib['size']
        for parent in child.iter(NS+'parent'):
            # Child depends on parent: edge from child to parent.
            comm_cost = parent.attrib.get('cost', None)
            if comm_cost is None:
                comm_cost = 0
                for output_file in child.iter(NS + 'uses'):
                    if output_file.attrib['link'] == 'output' and output_file.attrib['file'] in input_sizes:
                        comm_cost += input_sizes[output_file.attrib['file']]
            jobs[child.attrib['ref']].add_parent(parent.attrib['ref'], comm_cost)
            jobs[parent.attrib['ref']].add_child(child.attrib['ref'], comm_cost)
            parents[child.attrib['ref']].add((parent.attrib['ref'], comm_cost))
    print jobs
    return jobs, parents

###
class Docker(object):
     def __init__(self, attrib, tasks):
        super(Docker, self).__init__() #attrib of "object"
        self.tasks = tasks
        self.attrib = attrib
        if 'id' in attrib:
            self.id = attrib['id']
        if 'size' in attrib:
            self.size = attrib['size']
        if "trans_time" in attrib:
            self.trans_time = float(attrib['trans_time'])

     def __str__(self):
        return 'Docker{id: {id}}'.format(id=self.id)


def read_matrix(matrix,jobs): ##

    NS = '{http://pegasus.isi.edu/schema/DAX}'
    tree = ET.parse(matrix)
    root = tree.getroot()
    dockers={}


    for docker in root.iter(NS+"docker"): #need NS + job: element name
        tasks=[]
        for task in docker.iter(NS+"tasks"):
            tasks.append(task.text)
            for job in jobs.values():
                if task.text == job.id:
                    job.docker = Docker(docker.attrib, tasks) ##a docker object
        dockers[docker.attrib['id']] = Docker(docker.attrib, tasks) #set to an docker object


    print dockers
    return dockers, jobs
