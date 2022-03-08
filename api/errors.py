from flask import Flask, render_template, request, jsonify

from . import cors_headers


def custom_abort_page(message: str, code: int):
    return render_template("error_page.html", code=str(code), text=message)


def bind_error_pages(app: Flask):
    def error_func(err):
        if request.path.startswith("/api/v"):
            return jsonify(message=err.description), err.code, cors_headers
        return custom_abort_page(err.description, err.code), err.code, cors_headers

    for code in (500, 404, 400, 424, 409, 422, 403, 401):
        app.errorhandler(code)(error_func)