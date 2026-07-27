#!/usr/bin/env bash
# Shared fail-closed argument guards for Kaess launchers.


kaess_celsius_to_kelvin() {
  local python_bin="$1"
  local celsius="$2"

  "${python_bin}" -c \
    'import sys; print(273.15 + float(sys.argv[1]))' \
    "${celsius}"
}


kaess_validate_material_identity() {
  local repo_root="$1"
  local python_bin="$2"
  local material_config="$3"
  local parity_config="$4"
  local approval_record="$5"

  (
    cd "${repo_root}"
    PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${python_bin}" -m jax_fem_am.verification.material_identity \
      --material-config "${material_config}" \
      --parity-config "${parity_config}" \
      --approval-record "${approval_record}"
  )
}


kaess_validate_extra_value() {
  local option="$1"
  local value="$2"

  case "${option}" in
    --xla-pardiso-mode)
      case "${value}" in
        base|nocmp|cache-idx|phase23|fp32ir) ;;
        *)
          echo "kaess launcher: invalid ${option} value: ${value}" >&2
          return 2
          ;;
      esac
      ;;
    --summary-every)
      if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( 10#${value} < 1 )); then
        echo "kaess launcher: ${option} must be a positive integer" >&2
        return 2
      fi
      ;;
    --thermal-output-every|--mechanics-output-every)
      if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "kaess launcher: ${option} must be a non-negative integer" >&2
        return 2
      fi
      ;;
    --layers)
      if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( 10#${value} < 1 )); then
        echo "kaess launcher: ${option} must be a positive integer" >&2
        return 2
      fi
      ;;
  esac
}


kaess_parse_safe_extra_args() {
  local raw_args="${1:-}"
  local allow_layers="${2:-0}"
  local index=0
  local option=""
  local option_name=""
  local option_value=""

  KAESS_EXTRA_ARGV=()
  if [[ -n "${raw_args}" ]]; then
    read -r -a KAESS_EXTRA_ARGV <<< "${raw_args}"
  fi

  while (( index < ${#KAESS_EXTRA_ARGV[@]} )); do
    option="${KAESS_EXTRA_ARGV[index]}"
    case "${option}" in
      --mechanics-residual-only-check)
        index=$((index + 1))
        ;;
      --xla-pardiso-mode|--summary-every|--thermal-output-every|--mechanics-output-every)
        if (( index + 1 >= ${#KAESS_EXTRA_ARGV[@]} )) \
           || [[ "${KAESS_EXTRA_ARGV[index + 1]}" == --* ]]; then
          echo "kaess launcher: ${option} requires one value" >&2
          return 2
        fi
        kaess_validate_extra_value \
          "${option}" \
          "${KAESS_EXTRA_ARGV[index + 1]}" || return
        index=$((index + 2))
        ;;
      --xla-pardiso-mode=*|--summary-every=*|--thermal-output-every=*|--mechanics-output-every=*)
        option_name="${option%%=*}"
        option_value="${option#*=}"
        kaess_validate_extra_value "${option_name}" "${option_value}" || return
        index=$((index + 1))
        ;;
      --layers)
        if [[ "${allow_layers}" != "1" ]]; then
          echo "kaess launcher: unsupported EXTRA_ARGS option: ${option}" >&2
          return 2
        fi
        if (( index + 1 >= ${#KAESS_EXTRA_ARGV[@]} )) \
           || [[ "${KAESS_EXTRA_ARGV[index + 1]}" == --* ]]; then
          echo "kaess launcher: ${option} requires one value" >&2
          return 2
        fi
        kaess_validate_extra_value \
          "${option}" \
          "${KAESS_EXTRA_ARGV[index + 1]}" || return
        index=$((index + 2))
        ;;
      --layers=*)
        if [[ "${allow_layers}" != "1" ]]; then
          echo "kaess launcher: unsupported EXTRA_ARGS option: --layers" >&2
          return 2
        fi
        kaess_validate_extra_value "--layers" "${option#*=}" || return
        index=$((index + 1))
        ;;
      *)
        echo "kaess launcher: unsupported EXTRA_ARGS option: ${option}" >&2
        return 2
        ;;
    esac
  done
}


kaess_parse_safe_path_args() {
  local raw_args="${1:-}"
  local index=0
  local option=""

  KAESS_PATH_ARGV=()
  if [[ -n "${raw_args}" ]]; then
    read -r -a KAESS_PATH_ARGV <<< "${raw_args}"
  fi

  while (( index < ${#KAESS_PATH_ARGV[@]} )); do
    option="${KAESS_PATH_ARGV[index]}"
    case "${option}" in
      --layers|--layer-thickness|--power|--speed|--hatch|--sample-step|--jump-speed|--rotation-deg|--start-angle-deg)
        if (( index + 1 >= ${#KAESS_PATH_ARGV[@]} )) \
           || [[ "${KAESS_PATH_ARGV[index + 1]}" == --* ]]; then
          echo "kaess launcher: ${option} requires one value" >&2
          return 2
        fi
        index=$((index + 2))
        ;;
      --layers=*|--layer-thickness=*|--power=*|--speed=*|--hatch=*|--sample-step=*|--jump-speed=*|--rotation-deg=*|--start-angle-deg=*)
        index=$((index + 1))
        ;;
      *)
        echo "kaess launcher: unsupported PATH_ARGS option: ${option}" >&2
        return 2
        ;;
    esac
  done
}
