#!/usr/bin/env python
"""只定义模板类, 不存储数据. 类似C中的声明, 但不初始化"""
from hadron2np import methods


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        value = self[item]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


class Implementation:
    def __init__(self, name: str, quantity: str, function, arguments=None, **kwargs):
        """间接参数, 参数集合, 多种定义等需要计算得出的量, 使用此类来做中间信息存储.
        信息包括: 参考了谁的文章, 实现了多少参数(参数集合)等.
        TODO: 原则上 Implementation.function() 应该包含的是一个随机数的概率分布, 而不应该只是一个数(或是返回一个数的函数).

        Args:
            name:
            quantity: 被实现的量名
            function: 具体实现的函数 f(wc_obj, par_dict, *args, **kwargs)
            arguments: 指定实现函数所依赖的参数, 不包含在wc_obj和par_dict中.
            **kwargs: tex, description, cites
        """
        self.name = name
        self.quantity = quantity
        self.description = kwargs.get('description')
        self.cites = kwargs.get('cites')

        self.function = function
        self.arguments = arguments

        if 'container' in kwargs and isinstance(kwargs['container'], dict):
            kwargs['container'][self.name] = self
        self.attributions = kwargs

    def __repr__(self):
        return f'Implementation of Quantity("{self.quantity}"): {self.name}'

    def get_values(self, wc_obj, par_dict, *args, **kwargs) -> dict:
        """使用get方法取得Implementation类对象存储的数值."""

        # 如果实现函数依赖于自定义参数, 取值时判断是否给出了自定义参数
        if self.arguments is not None:
            for item in self.arguments:
                if item not in kwargs:
                    raise ValueError(f'Quantity({self.quantity}) depends on: {item}')

        return self.function(wc_obj, par_dict, *args, **kwargs)


class Quantity:
    def __init__(self, name, **kwargs):
        """参数除了数值, 还会有一些描述信息, 使用此类来做中间信息存储.

        Args:
            name:
            **kwargs: tex, description, cites
        """
        self.name = name
        self.tex = kwargs.get('tex', name)
        self.description = kwargs.get('description')

        if 'container' in kwargs and isinstance(kwargs['container'], dict):
            kwargs['container'][self.name] = self
        self.attributions = kwargs

        self.implementations = {}  # 一个量的数值不直接给出, 由Implementation中间类来存储.

    def __repr__(self) -> str:
        return f'Quantity("{self.name}")'

    def add_implementation(self, i_obj: Implementation):
        self.implementations[i_obj.name] = i_obj

    def get_implementation(self, **kwargs):
        try:
            i_name = kwargs.get('name', next(iter(self.implementations)))
            return self.implementations[i_name]
        except (KeyError, StopIteration):
            raise KeyError("No implementation specified for quantity " + self.name)

    def show_implementations(self):
        all_dict = {}
        for i_name, i_inst in self.implementations.items():
            all_dict[i_name] = i_inst.description
        return all_dict

    # def prediction_central(self, constraints_obj, wc_obj, *args, **kwargs):
    #     implementation = self.get_implementation()
    #     return implementation.get_central(constraints_obj, wc_obj, *args, **kwargs)

    # def prediction(self, par_dict, wc_obj, *args, **kwargs):
    #     implementation = self.get_implementation()
    #     return implementation.get(par_dict, wc_obj, *args, **kwargs)


class ParameterGroup(Quantity):
    def __init__(self, name, members: list[str], **kwargs):
        """

        Args:
            name:
            members: 包含的子元素
            **kwargs:
        """
        # self.name = name
        # self.tex = kwargs.get('tex', name)
        # self.description = kwargs.get('description')

        if 'container' in kwargs and isinstance(kwargs['container'], dict):
            kwargs['container'][name] = self
            kwargs.pop('container')
        super().__init__(name, **kwargs)

        self.members = members
        self.attributions = kwargs
        self.implementations = {}  # 一个量的数值不直接给出, 由Implementation中间类来存储.

    def __repr__(self) -> str:
        return f'QuantityGroup("{self.name}"): {self.members}'

    # def get_central(self, wc_obj, par_dict, imp_name=None, **kwargs) -> dict:
    #     imp: Implementation = self.get_implementation()
    #     res = imp.get_central(wc_obj=None, par_dict=None)
    #     return res
        for member in self.members:
            # TODO: 这里没有容器, 所以没有办法存储Group中的单个Quantity
            # 只能在初始化时才
            pass
            # Quantity()


class Parameter(Quantity):
    """直接参数: 定义值是一个常数(加误差), 不依赖于其它参数.
    通常只有一种实现, 即直接给定数值. """
    def __init__(self, name, **kwargs):
        if 'container' in kwargs and isinstance(kwargs['container'], dict):
            kwargs['container'][name] = self
            kwargs.pop('container')
        super().__init__(name, **kwargs)
        self.implementation = None

    def set_value(self, value):
        self.implementation = Implementation(name=self.name+' value',
                                             quantity=self.name,
                                             function=lambda wc, par: methods.parse_errors(value))
        self.add_implementation(self.implementation)

    def get_central(self):
        if self.implementation is None:
            self.implementation = self.get_implementation()
        return self.implementation.get_values(None, None)['central']


class Observable(Quantity):
    """可观测量: 定义为理论的计算解析表达式. 它还要联系上实验观测."""

    def __init__(self, name, **kwargs):
        if 'container' in kwargs and isinstance(kwargs['container'], dict):
            kwargs['container'][name] = self
            kwargs.pop('container')
        super().__init__(name)

    def __repr__(self) -> str:
        return f'Observable("{self.name}")'


class Measurement(Quantity):
    def __init__(self, name, observable, **kwargs):
        if 'container' in kwargs and isinstance(kwargs['container'], dict):
            kwargs['container'][name] = self
            kwargs.pop('container')
        if 'value' in kwargs.keys():
            pass
        super().__init__(name, **kwargs)
        self.observable = observable
