import copy


def __is_packable_undefined_warning(warn):
    return "Potential undefined types found" in warn.message and not warn.packed


def __same_line_and_same_stack_trace_warning(warn1, warn2):
    if warn2.packed or warn1.packed:
        return False

    if warn1.localization.line != warn2.localization.line:
        return False

    if str(warn1.stack_trace_snapshot) != str(warn2.stack_trace_snapshot):
        return False

    return True


def __unify_undefined_warnings(warn1, warn2):
    if warn1 is None:
        return warn2

    packed_warning = copy.copy(warn1)
    packed_warning.message = "Potential multiple undefined types found in this line"
    if hasattr(packed_warning.localization, 'column_offsets_for_packed_warnings'):
        packed_warning.localization.column_offsets_for_packed_warnings.append(warn2.localization.column)
    else:
        packed_warning.localization.column_offsets_for_packed_warnings = [warn2.localization.column]

    packed_warning.rebuild_message()
    warn1.packed = True
    warn2.packed = True

    return packed_warning


def __unify_same_line_warnings(warn1, warn2):
    if warn1 is None:
        return warn2

    packed_warning = copy.copy(warn1)
    if warn1.message != warn2.message:
        packed_warning.message = warn1.message + "\n\t" + warn2.message
    else:
        packed_warning.message = warn1.message

    if hasattr(packed_warning.localization, 'column_offsets_for_packed_warnings'):
        packed_warning.localization.column_offsets_for_packed_warnings.append(warn2.localization.column)
    else:
        packed_warning.localization.column_offsets_for_packed_warnings = [warn2.localization.column]

    packed_warning.rebuild_message()
    warn1.packed = True
    warn2.packed = True

    return packed_warning


def pack_undefined_warnings(TypeWarning):
    """This method consolidates multiple warnings into one provided the following conditions are met:
    1) Belong to the same line
    2) Has a message that refer to potential undefined types
    3) Has the same call stack.

    In that case, a single warning is produced, and multiple columns are stored to indicate all the places in the
    line that may present this warning. This greatly helps to lower the amount of reported warnings produced when
    multiple arithmetic operations deal with operands with potential UndefinedType values.
    """
    packables = filter(lambda w: __is_packable_undefined_warning(w), TypeWarning.warnings)
    non_packables = filter(lambda w: not __is_packable_undefined_warning(w), TypeWarning.warnings)
    packeds_temp = []
    packeds = []

    for pw in packables:
        if pw.packed:
            continue
        packed_warning = pw
        for w in TypeWarning.warnings:
            if pw == w:
                continue
            if w.packed:
                continue
            if __is_packable_undefined_warning(w):
                # Can we pack these warnings
                if __same_line_and_same_stack_trace_warning(pw, w):
                    packeds_temp.append(w)

        # Pack all warnings into one
        for p in packeds_temp:
            packed_warning = __unify_undefined_warnings(packed_warning, p)

        packeds_temp = []
        if packed_warning not in packeds:
            packeds.append(packed_warning)

    TypeWarning.warnings = non_packables + packeds

    def unpack(warn):
        warn.packed = False
        return warn

    TypeWarning.warnings = map(lambda warn: unpack(warn), TypeWarning.warnings)


def pack_warnings_with_the_same_line_and_stack_trace(TypeWarning):
    """This method consolidates multiple warnings into one provided the following conditions are met:
    1) Belong to the same line
    2) Has different messages
    3) Has the same call stack.

    In that case, a single warning is produced, and multiple columns are stored to indicate all the places in the
    line that may present this warning. This greatly helps to lower the amount of reported warnings produced when
    multiple arithmetic operations deal with operands with potential UndefinedType values.
    """
    packeds_temp = []
    packeds = []

    for pw in TypeWarning.warnings:
        if pw.packed:
            continue
        packed_warning = pw
        for w in TypeWarning.warnings:
            if pw == w:
                continue
            if w.packed:
                continue
            # Can we pack these warnings
            if __same_line_and_same_stack_trace_warning(pw, w):
                packeds_temp.append(w)

        # Pack all warnings into one
        for p in packeds_temp:
            packed_warning = __unify_same_line_warnings(packed_warning, p)

        packeds_temp = []
        if not packed_warning in packeds:
            packeds.append(packed_warning)

    TypeWarning.warnings = packeds


def __same_line_and_message_but_different_stack_trace_warning(warn1, warn2):
    if warn2.packed or warn1.packed:
        return False

    if warn1.localization.line != warn2.localization.line:
        return False

    if warn1.get_display_message() != warn2.get_display_message():
        return False

    if str(warn1.stack_trace_snapshot) != str(warn2.stack_trace_snapshot):
        return True

    return False


def __unify_same_line_and_message_but_different_stack_trace_warning(warn1, warn2):
    if warn1 is None:
        return warn2

    packed_warning = copy.copy(warn1)
    packed_warning.other_stack_traces.append(warn2.stack_trace_snapshot)

    packed_warning.rebuild_message()
    warn1.packed = True
    warn2.packed = True

    return packed_warning


def pack_warnings_with_the_same_line_and_message_but_different_stack_trace(TypeWarning):
    """This method consolidates multiple warnings into one provided the following conditions are met:
    1) Belong to the same line
    2) Has different messages
    3) Has the same call stack.

    In that case, a single warning is produced, and multiple columns are stored to indicate all the places in the
    line that may present this warning. This greatly helps to lower the amount of reported warnings produced when
    multiple arithmetic operations deal with operands with potential UndefinedType values.
    """
    packeds_temp = []
    packeds = []

    for pw in TypeWarning.warnings:
        if pw.packed:
            continue
        packed_warning = pw
        for w in TypeWarning.warnings:
            if pw == w:
                continue
            if w.packed:
                continue
            # Can we pack these warnings
            if __same_line_and_message_but_different_stack_trace_warning(pw, w):
                packeds_temp.append(w)

        # Pack all warnings into one
        for p in packeds_temp:
            packed_warning = __unify_same_line_and_message_but_different_stack_trace_warning(packed_warning, p)

        packeds_temp = []
        if not packed_warning in packeds:
            packeds.append(packed_warning)

    TypeWarning.warnings = packeds


def get_call_stack_str(stack_trace_snap, additional_snaps):
    if len(additional_snaps) == 0:
        return str(stack_trace_snap)
    else:
        temp = "Multiple call sources:\n(1) " + str(stack_trace_snap)
        counter = 2
        for snap in additional_snaps:
            temp += "\n(" + str(counter) + ") " + str(snap)
            counter+=1

        return temp
