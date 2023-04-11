rule test_run:
     log: 'test_run.log'
     run:
        import numpy
        print("Success!")

rule test_shell:
     log: 'test_shell.log'
     shell:
        "python -c 'import numpy'; echo success!"