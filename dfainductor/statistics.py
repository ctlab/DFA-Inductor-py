from timeit import default_timer as timer

from typing import Dict, List

from .logging import log_time, log_statistics, log_br


class STATISTICS:
    APTA = 'APTA building'
    FORMULA = 'Formula building'
    FEEDING = 'Solver feeding'
    SOLVING = 'SAT solving'
    WHOLE = 'Whole task'

    times: Dict[str, List[float]] = {}
    timers: Dict[str, float] = {}

    @classmethod
    def start_timer(cls, name: str) -> None:
        if name not in cls.times.keys():
            cls.times[name] = []
            cls.timers[name] = 0
        if cls.timers[name] != 0:
            raise TimerWasNotStoppedBeforeNewStart(name)
        cls.timers[name] = timer()

    @classmethod
    def stop_timer(cls, name: str) -> float:
        if name not in cls.times.keys() or cls.timers[name] == 0:
            raise TimerWasNotStartedBeforeStopped(name)
        cls.times[name].append(timer() - cls.timers[name])
        cls.timers[name] = 0
        log_time(name + 'time', cls.times[name][-1])
        return cls.times[name][-1]

    @classmethod
    def print_statistics(cls):
        log_br()
        for name, list_of_times in cls.times.items():
            log_statistics(name, sum(list_of_times))
        log_br()

    @classmethod
    def start_apta_building_timer(cls) -> None:
        cls.start_timer(cls.APTA)

    @classmethod
    def stop_apta_building_timer(cls) -> float:
        return cls.stop_timer(cls.APTA)

    @classmethod
    def start_formula_timer(cls) -> None:
        cls.start_timer(cls.FORMULA)

    @classmethod
    def stop_formula_timer(cls) -> float:
        return cls.stop_timer(cls.FORMULA)

    @classmethod
    def start_feeding_timer(cls) -> None:
        cls.start_timer(cls.FEEDING)

    @classmethod
    def stop_feeding_timer(cls) -> float:
        return cls.stop_timer(cls.FEEDING)

    @classmethod
    def start_solving_timer(cls) -> None:
        cls.start_timer(cls.SOLVING)

    @classmethod
    def stop_solving_timer(cls) -> float:
        return cls.stop_timer(cls.SOLVING)

    @classmethod
    def start_whole_timer(cls) -> None:
        cls.start_timer(cls.WHOLE)

    @classmethod
    def stop_whole_timer(cls) -> float:
        return cls.stop_timer(cls.WHOLE)


class TimerWasNotStoppedBeforeNewStart(Exception):
    pass


class TimerWasNotStartedBeforeStopped(Exception):
    pass
