import pathlib
import os
import shutil
import matplotlib.pyplot
import numpy as np


class Helper():
    def __init__(self):
        pass

    def create_dir(self,
                   path_dir: str,
                   overwrite: bool = False,
                   add_missing_parent_dirs: bool = True,
                   raise_exception: bool = False,
                   print_message: bool = False) \
            -> tuple[str, bool]:
        return self._create_dir(path_dir, overwrite, add_missing_parent_dirs, raise_exception, print_message)

    @staticmethod
    def _create_dir(target: str,
                    overwrite: bool,
                    add_missing_parent_dirs: bool,
                    raise_exception: bool,
                    print_message: bool) -> tuple[str, bool]:
        msg, keep_going = str(), bool()
        try:
            if overwrite:
                if os.path.isdir(target):
                    shutil.rmtree(target)
                    msg = f"Existing directory {target} was overwritten."
                else:
                    msg = f"Could not overwrite {target} as it did not exist. Created it instead."
                keep_going = True
            else:
                msg, keep_going = f"Directory {target} created successfully.", True
            pathlib.Path(target).mkdir(parents=add_missing_parent_dirs, exist_ok=False)
        except Exception as exc:
            if exc.args[0] == 2:  # FileNotFoundError
                if raise_exception:
                    raise FileNotFoundError(f"Not all parent directories exist for directory {target}.")
                else:
                    msg, keep_going = f"Not all parent directories exist for directory {target}.", False
            elif exc.args[0] == 17:  # FileExistsError
                if raise_exception:
                    raise FileExistsError(f"Directory {target} already exists and was not changed.")
                else:
                    msg, keep_going = f"Directory {target} already exists and was not changed.", False
        if print_message:
            print(msg)
        return msg, keep_going

    @staticmethod
    def handle_figure(figure: matplotlib.pyplot.figure,
                      save_to: str=False,
                      show: bool=False,
                      size: tuple=(18.5, 10),
                      inches: int=100,
                      tight_layout: bool=True,
                      close: bool=True) -> matplotlib.pyplot.figure:
        figure.set_size_inches(size)
        figure.set_dpi(inches)
        figure.set_tight_layout(tight_layout)
        if save_to:
            figure.savefig(save_to)
        if show:
            matplotlib.pyplot.show()
        if close and not show:
            matplotlib.pyplot.close(figure)
        else:
            return figure

    def handle_axis(self,
                    axis: matplotlib.pyplot.axis or np.ndarray[matplotlib.pyplot.axis],
                    title: str or list[str]=None,
                    grid: bool or list[bool]=True,
                    legend: bool=True,
                    legend_loc: int=0,
                    legend_columns: int=1,
                    legend_together: int=False,
                    x_label: str or list[str]=None,
                    y_label: str or list[str]=None,
                    z_label: str=None,
                    x_scale: str="linear",
                    y_scale: str="linear",
                    x_lim: tuple=None,
                    y_lim: tuple=None,
                    font_size: int=20,
                    line_width: int=3,
                    label_pad: float=5) -> matplotlib.pyplot.axis or np.ndarray[matplotlib.pyplot.axis]:
        """
        This function does a shit ton for convenience (which also means perhaps something against your will). Most
        things should be straight forward (such as line_width and font_size), some will be explained in a second,
        the rest is probably not needed for our plots.
        How to use 'title', 'x_label', and 'y_label': Use a single string if every axis should get the same string.
        Use ['example_string'] if only the first axis should get the string. Use ['string_1', 'string_2',
        ...] if every axis should get a different string. If you want skip an axis, use None (not as string). If you
        have a grid of axes, the indexing will be done row wise (first through the first row, then the second, ...)
        The same principle holds for 'grid' (just with boolean values instead of strings).
        If there's issues text me (Jonas).
        @param axis:
        @param title:
        @param grid:
        @param legend:
        @param legend_loc:
        @param legend_columns:
        @param x_label:
        @param y_label:
        @param z_label:
        @param x_scale:
        @param y_scale:
        @param x_lim:
        @param y_lim:
        @param font_size:
        @param line_width:
        @param label_pad:
        @return:
        """
        if type(axis) != np.ndarray:
            axis, shape = [axis], None
        else:
            shape = axis.shape
            axis = axis if len(shape) == 1 else [ax for ax in axis.flatten()]
        title, x_label, y_label, grid = self._fill_arrays(len(axis), title=title, x_label=x_label, y_label=y_label,
                                                          grid=grid)
        for i, ax in enumerate(axis):
            ax.set_title(title[i])

            if x_lim is not None: ax.set_xlim(x_lim)
            if y_lim is not None: ax.set_ylim(y_lim)

            if font_size:
                ax.title.set_fontsize(font_size)
                ax.xaxis.label.set_fontsize(font_size)
                ax.xaxis.get_offset_text().set_fontsize(font_size)
                ax.yaxis.label.set_fontsize(font_size)
                ax.yaxis.get_offset_text().set_fontsize(font_size)
                if ax.name == "3d":
                    ax.zaxis.label.set_fontsize(font_size)
                ax.tick_params(axis='both', labelsize=font_size)

            ax.grid(grid[i])
            ax.set_xlabel(x_label[i], labelpad=label_pad)
            ax.set_ylabel(y_label[i], labelpad=label_pad)
            if z_label is not None:
                ax.set_zlabel(z_label, labelpad=label_pad)

            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)

            if legend or line_width:
                if (type(legend_together) == int or line_width) and i==0:
                    lines = list()
                    for axs in axis:
                        lines += axs.get_lines()
                    if line_width:
                        for line in lines:
                            line.set_linewidth(line_width)
                    labels = [line.get_label() for line in lines if line.get_label()[0] != "_"]
                    if type(legend_together) == int:
                        if font_size:
                            axis[legend_together].legend(lines, labels, ncol=legend_columns, prop={"size": font_size},
                                                         loc=legend_loc)
                        else:
                            axis[legend_together].legend(lines, labels, ncol=legend_columns, loc=legend_loc)
                if not legend_together:
                    lines = ax.get_lines()
                    lines = [line for line in lines if line.get_label()[0] != "_"]
                    labels = [line.get_label() for line in lines if line.get_label()[0] != "_"]
                    if legend:
                        if font_size:
                            ax.legend(lines, labels, ncol=legend_columns, prop={"size": font_size}, loc=legend_loc)
                        else:
                            ax.legend(lines, labels, ncol=legend_columns, loc=legend_loc)
        return axis if shape is None else np.asarray(axis).reshape(shape)

    @staticmethod
    def _fill_arrays(length: int, **kwargs):
        filled = list()
        for parameter, argument in kwargs.items():
            fill_with = None if parameter != "grid" else False
            if type(argument) == list:
                if len(argument) == 1:
                    new = argument+[fill_with for _ in range(length-1)]
                elif len(argument) == length:
                    new = argument
                else:
                    raise ValueError(f"'{parameter}' was tried to be set for each axis. However, {len(argument)} "
                                     f"values were supplied while {length} axes exist.")
            elif type(argument) == str:
                new = [argument for _ in range(length)]
            elif argument is None:
                new = [fill_with for _ in range(length)]
            elif argument:
                new = [argument for _ in range(length)]
            elif parameter == "grid":
                new = [fill_with for _ in range(length)]
            else:
                raise ValueError(f"Check yo inputs for parameter {parameter}.")
            filled.append(new)
        return filled
