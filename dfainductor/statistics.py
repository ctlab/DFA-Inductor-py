from timeit import default_timer as timer
from typing import Dict

from .logging import log_time, log_statistics, log_br


# noinspection DuplicatedCode
class STATISTICS:
    APTA_NAME = 'APTA building'
    APTA_SUM = 0
    APTA_CUR = 0

    IG_NAME = 'IG building'
    IG_SUM = 0
    IG_CUR = 0

    FORMULA_NAME = 'Formula building'
    FORMULA_SUM = 0
    FORMULA_CUR = 0

    SOLVING_NAME = 'SAT solving'
    SOLVING_SUM = 0
    SOLVING_CUR = 0

    WHOLE_NAME = 'Whole task'
    WHOLE_SUM = 0
    WHOLE_CUR = 0

    times_sum: Dict[str, float] = {}
    times_current: Dict[str, float] = {}

    @classmethod
    def start_timer(cls, name: str) -> None:
        if name not in cls.times_sum.keys():
            cls.times_sum[name] = 0
            cls.times_current[name] = 0
        if cls.times_current[name] != 0:
            raise TimerWasNotStoppedBeforeNewStart(name)
        cls.times_current[name] = timer()

    @classmethod
    def stop_timer(cls, name: str) -> float:
        end_time = timer()
        total_time = end_time - cls.times_current[name]
        if name not in cls.times_sum.keys() or cls.times_current[name] == 0:
            raise TimerWasNotStartedBeforeStopped(name)
        log_time(name + 'time', total_time)
        cls.times_sum[name] += total_time
        cls.times_current[name] = 0
        return total_time

    @classmethod
    def print_statistics(cls):
        log_br()
        log_statistics(cls.APTA_NAME, cls.APTA_SUM)
        log_statistics(cls.IG_NAME, cls.IG_SUM)
        log_statistics(cls.FORMULA_NAME, cls.FORMULA_SUM)
        log_statistics(cls.SOLVING_NAME, cls.SOLVING_SUM)
        log_statistics(cls.WHOLE_NAME, cls.WHOLE_SUM)
        for name, list_of_times in cls.times_sum.items():
            log_statistics(name, list_of_times)
        log_br()

    @classmethod
    def start_apta_building_timer(cls) -> None:
        if cls.APTA_CUR != 0:
            raise TimerWasNotStoppedBeforeNewStart(cls.APTA_NAME)
        cls.APTA_CUR = timer()

    @classmethod
    def stop_apta_building_timer(cls) -> float:
        end_time = timer()
        total_time = end_time - cls.APTA_CUR
        if cls.APTA_CUR == 0:
            raise TimerWasNotStartedBeforeStopped(cls.APTA_NAME)
        log_time(cls.APTA_NAME + 'time', total_time)
        cls.APTA_SUM += total_time
        cls.APTA_CUR = 0
        return total_time

    @classmethod
    def start_ig_building_timer(cls) -> None:
        if cls.IG_CUR != 0:
            raise TimerWasNotStoppedBeforeNewStart(cls.IG_NAME)
        cls.IG_CUR = timer()

    @classmethod
    def stop_ig_building_timer(cls) -> float:
        end_time = timer()
        total_time = end_time - cls.IG_CUR
        if cls.IG_CUR == 0:
            raise TimerWasNotStartedBeforeStopped(cls.IG_NAME)
        log_time(cls.IG_NAME + 'time', total_time)
        cls.IG_SUM += total_time
        cls.IG_CUR = 0
        return total_time

    @classmethod
    def start_formula_timer(cls) -> None:
        if cls.FORMULA_CUR != 0:
            raise TimerWasNotStoppedBeforeNewStart(cls.FORMULA_NAME)
        cls.FORMULA_CUR = timer()

    @classmethod
    def stop_formula_timer(cls) -> float:
        end_time = timer()
        total_time = end_time - cls.FORMULA_CUR
        if cls.FORMULA_CUR == 0:
            raise TimerWasNotStartedBeforeStopped(cls.FORMULA_NAME)
        log_time(cls.FORMULA_NAME + 'time', total_time)
        cls.FORMULA_SUM += total_time
        cls.FORMULA_CUR = 0
        return total_time

    @classmethod
    def start_solving_timer(cls) -> None:
        if cls.SOLVING_CUR != 0:
            raise TimerWasNotStoppedBeforeNewStart(cls.SOLVING_NAME)
        cls.SOLVING_CUR = timer()

    @classmethod
    def stop_solving_timer(cls) -> float:
        end_time = timer()
        total_time = end_time - cls.SOLVING_CUR
        if cls.SOLVING_CUR == 0:
            raise TimerWasNotStartedBeforeStopped(cls.SOLVING_NAME)
        log_time(cls.SOLVING_NAME + 'time', total_time)
        cls.SOLVING_SUM += total_time
        cls.SOLVING_CUR = 0
        return total_time

    @classmethod
    def start_whole_timer(cls) -> None:
        if cls.WHOLE_CUR != 0:
            raise TimerWasNotStoppedBeforeNewStart(cls.WHOLE_NAME)
        cls.WHOLE_CUR = timer()

    @classmethod
    def stop_whole_timer(cls) -> float:
        end_time = timer()
        total_time = end_time - cls.WHOLE_CUR
        if cls.WHOLE_CUR == 0:
            raise TimerWasNotStartedBeforeStopped(cls.WHOLE_NAME)
        log_time(cls.WHOLE_NAME + 'time', total_time)
        cls.WHOLE_SUM += total_time
        cls.WHOLE_CUR = 0
        return total_time


class TimerWasNotStoppedBeforeNewStart(Exception):
    pass


class TimerWasNotStartedBeforeStopped(Exception):
    pass
