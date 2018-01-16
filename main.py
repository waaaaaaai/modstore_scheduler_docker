#!/usr/bin/env python
import dax
import cpf
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dax_file", default="data/Epigenomics_24.xml")
    args = parser.parse_args()
    jobs, parents = dax.read_file(args.dax_file)
    dockers,jobs= dax.read_matrix("data/new_form_input.xml", jobs)
    scheduler = cpf.CriticalPathFirst(jobs, dockers)
    print scheduler.compute()
    print "a"

if __name__ == '__main__':
    main()
