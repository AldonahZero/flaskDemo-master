from flask_restx import fields

# from . import create_id_field, ModelCreator
from config.setting import MAX_FILE_SIZE
from . import ModelCreator

class MulModel(ModelCreator):
    # id = create_id_field(
    #     required=True,
    #     description="Attachment ID in database",
    # )
    # post = create_id_field(
    #     required=True,
    #     description="ID of Post that contains this attachment"
    # )
    filename = fields.String(
        required=True,
        description="Name of an attachment",
        example="gravity_gun_firmware.bin",
        min_length=1
    )
    size = fields.Integer(
        required=True,
        description="Size of attachment (in bytes)",
        example=256256,
        min=0,
        max=MAX_FILE_SIZE
    )