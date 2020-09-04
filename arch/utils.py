import os
import torch
import copy


def output_csv(the_path, data_dict, order=None, delimiter=','):
    if the_path.endswith('.tsv'):
        delimiter = '\t'

    is_file_exists = os.path.exists(the_path)
    with open(the_path, 'a+') as op:
        keys = list(data_dict.keys())
        if order is not None:
            keys = order + [k for k in keys if k not in order]

        col_title = delimiter.join([str(k) for k in keys])
        if not is_file_exists:
            print(col_title, file=op)
        else:
            old_col_title = open(the_path, 'r').readline().strip()
            if col_title != old_col_title:
                old_order = old_col_title.split(delimiter)

                no_key = [k for k in old_order if k not in keys]
                if len(no_key) > 0:
                    raise (RuntimeError('The data_dict does not have the '
                                        'following old keys: %s' % str(no_key)))

                additional_keys = [k for k in keys if k not in old_order]
                if len(additional_keys) > 0:
                    print('WARNING! The data_dict has following additional '
                          'keys %s.' % (str(additional_keys)))
                    col_title = delimiter.join([
                        str(k) for k in old_order + additional_keys])
                    print(col_title, file=op)

                keys = old_order + additional_keys

        vals = []
        for k in keys:
            val = data_dict[k]
            if isinstance(val, torch.Tensor) and val.ndim == 0:
                val = val.item()
            vals.append(str(val))

        print(delimiter.join(vals), file=op)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, DotDict):
                self[k] = DotDict(v)

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(dict(self), memo=memo))
