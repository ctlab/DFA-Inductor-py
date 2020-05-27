from pysat.formula import IDPool


class VarPool:

    def __init__(self) -> None:
        self._vpool = IDPool()

    def var(self, name: str, ind1, ind2=0, ind3=0) -> int:
        return self._vpool.id(f'{name}_{ind1}_{ind2}_{ind3}')

    def var_name(self, id_: int):
        return self._vpool.obj(id_)
