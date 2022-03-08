from typing import Dict, Generator, Set

from flask_restx import fields
from flask import request


ID_EXAMPLE = 'd1d3ee42-731c-04d9-0eee-16d3e7a62948'
EMAIL_EXAMPLE = 'gordonfreeman@city17.net'
EMAIL_PATTERN = r'\S+@\S+\.\S+'
PASSWORD_EXAMPLE = 'Qwerty123'
DATETIME_EXAMPLE = '2019-08-18T13:41:05'
DATETIME_FORMAT = "iso8601"  # was 'rfc822' in example


class ModelCreator:
    def __new__(cls, *args, **kwargs):
        result = {}
        raw_result = dir(cls)
        for item in raw_result:
            if not item.startswith("__") or not item.endswith("__"):  # if item is not like __****__
                result[item] = getattr(cls, item)
        return result


def copy_field(field, required: bool = None, description: str = None):
    result = type(field)(**field.__dict__)
    if required is not None:
        result.required = required
    if description is not None:
        result.description = description
    return result


def create_id_field(required=False, description=""):
    return fields.String(
        required=required,
        description=description,
        example=ID_EXAMPLE,
        min_length=36,
        max_length=36
    )


def create_email_field(required=False, description=""):
    return fields.String(
        required=required,
        description=description,
        example=EMAIL_EXAMPLE,
        min_length=5,
        max_length=256
    )


def create_datetime_field(required=False, description=""):
    return fields.DateTime(
        required=required,
        description=f"{description} (the format is '{DATETIME_FORMAT}')",
        example=DATETIME_EXAMPLE,
        dt_format=DATETIME_FORMAT
    )


pages_count_model = fields.Integer(
    required=False,
    description="Total count of pages in the resource",
    example=1,
    min=0
)


def required_query_params(request_args: Dict[str, str or Dict[str, str]]) -> Dict[str, Dict[str, str or bool]]:
    result = {}
    for item in request_args:
        if isinstance(request_args[item], str):
            result[item] = {"description": request_args[item], "required": True}
        elif isinstance(request_args[item], dict):
            if not result.get(item):
                result[item] = request_args[item]
            result[item]['required'] = True
    return result


def update_dict(dict1: dict or ModelCreator, dict2: dict or ModelCreator) -> dict:
    result = dict1.copy()
    result.update(dict2)
    return result


def iterate_query_param(param_value: str) -> Generator[str, None, None]:
    result_raw = param_value.replace(' ', '').strip(',').split(',')
    for item in result_raw:
        if item:
            yield item


def query_param_to_set(param_name: str) -> Set[str]:
    return set(iterate_query_param(request.args.get(param_name, '')))