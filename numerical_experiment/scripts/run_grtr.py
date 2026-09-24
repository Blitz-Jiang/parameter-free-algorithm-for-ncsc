if __package__:
    from .run_comparision import main
else:
    from run_comparision import main

if __name__ == '__main__':
    main(default_algorithms=('grtr',))
