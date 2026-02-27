"""Run evaluation pipeline for MidPrj using CLI parameters."""
# -*- coding: utf-8 -*-
from . import midprj_main as prj
from . import midprj_eval
from call_midprj_args import build_argument_parser, build_param_from_args


def main():
	parser = build_argument_parser(description="MidPrj RAG 평가 실행 스크립트")
	args = parser.parse_args()
	param = build_param_from_args(args)
	if  param.do_eval == "optimize":
		prj.Execute_eval_optimize(param)
	elif param.do_eval == "all":
		prj.Execute_evalEx(param)
	else:
		prj.Execute_eval(param)
if __name__ == "__main__":
	main()
