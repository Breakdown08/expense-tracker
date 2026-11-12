"""The rendered overlay `k8s/overlays/k3d`, checked without a cluster."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT: Path = Path(__file__).resolve().parents[1]
NS: str = "expense-tracker"
SECRET: str = "expense-tracker-env"

type Doc = dict[str, Any]


def need(binary: str) -> None:
    if shutil.which(binary) is None:
        if os.environ.get("CI"):
            pytest.fail(f"{binary} is not in PATH, but CI is set")
        pytest.skip(f"{binary} is not in PATH")


def render_result() -> subprocess.CompletedProcess[str]:
    need("kubectl")
    return subprocess.run(
        ["kubectl", "kustomize", "k8s/overlays/k3d"], cwd=ROOT, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="session")
def docs() -> list[Doc]:
    result: subprocess.CompletedProcess[str] = render_result()
    assert result.returncode == 0, result.stderr
    return [d for d in yaml.safe_load_all(result.stdout) if d]


@pytest.fixture(scope="session")
def postgres_image() -> str:
    compose: Doc = yaml.safe_load((ROOT / "compose.yaml").read_text())
    image: str = compose["services"]["db"]["image"]
    return image


def one(docs: list[Doc], kind: str, name: str) -> Doc:
    found: list[Doc] = [d for d in docs if d["kind"] == kind and d["metadata"]["name"] == name]
    assert len(found) == 1, f"{kind}/{name}: {len(found)} found"
    return found[0]


def configmap(docs: list[Doc]) -> Doc:
    found: list[Doc] = [
        d for d in docs if d["kind"] == "ConfigMap" and d["metadata"]["name"].startswith("pg-backup-")
    ]
    assert len(found) == 1
    return found[0]


def pod_specs(docs: list[Doc]) -> list[tuple[str, Doc]]:
    specs: list[tuple[str, Doc]] = []
    d: Doc
    for d in docs:
        if d["kind"] in ("Deployment", "StatefulSet"):
            specs.append((d["kind"], d["spec"]["template"]["spec"]))
        elif d["kind"] == "CronJob":
            specs.append((d["kind"], d["spec"]["jobTemplate"]["spec"]["template"]["spec"]))
    return specs


def all_containers(docs: list[Doc]) -> list[Doc]:
    out: list[Doc] = []
    out += [c for _, spec in pod_specs(docs) for c in spec.get("initContainers", []) + spec["containers"]]
    return out


def container(spec: Doc, name: str) -> Doc:
    found: list[Doc] = [c for c in spec.get("initContainers", []) + spec["containers"] if c["name"] == name]
    assert len(found) == 1
    return found[0]


def bot_spec(docs: list[Doc]) -> Doc:
    spec: Doc = one(docs, "Deployment", "bot")["spec"]["template"]["spec"]
    return spec


def pg_spec(docs: list[Doc]) -> Doc:
    spec: Doc = one(docs, "StatefulSet", "postgres")["spec"]["template"]["spec"]
    return spec


def cron_spec(docs: list[Doc]) -> Doc:
    spec: Doc = one(docs, "CronJob", "pg-backup")["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    return spec


def secret_key(password: Doc) -> Doc:
    ref: Doc = password["valueFrom"]["secretKeyRef"]
    return ref


def test_kustomize_renders() -> None:
    result: subprocess.CompletedProcess[str] = render_result()
    assert result.returncode == 0, result.stderr


def test_rendered_objects_are_exactly_the_expected_set(docs: list[Doc]) -> None:
    got: set[tuple[str, str]] = {
        (d["kind"], "pg-backup-" if d["kind"] == "ConfigMap" else d["metadata"]["name"]) for d in docs
    }
    assert len(docs) == 7
    assert got == {
        ("Namespace", NS),
        ("Service", "postgres"),
        ("StatefulSet", "postgres"),
        ("Deployment", "bot"),
        ("PersistentVolumeClaim", "backups"),
        ("CronJob", "pg-backup"),
        ("ConfigMap", "pg-backup-"),
    }
    assert configmap(docs)["metadata"]["name"] != "pg-backup-"
    assert not [d for d in docs if d["kind"] == "Secret"]


def test_everything_but_namespace_lives_in_expense_tracker(docs: list[Doc]) -> None:
    others: list[Doc] = [d for d in docs if d["kind"] != "Namespace"]
    assert others
    assert all(d["metadata"].get("namespace") == NS for d in others)


def test_bot_is_single_and_recreated(docs: list[Doc]) -> None:
    spec: Doc = one(docs, "Deployment", "bot")["spec"]
    assert spec["replicas"] == 1
    assert spec["strategy"]["type"] == "Recreate"


def test_bot_init_containers_and_commands(docs: list[Doc], postgres_image: str) -> None:
    spec: Doc = bot_spec(docs)
    assert [c["name"] for c in spec["initContainers"]] == ["wait-postgres", "migrate"]
    assert container(spec, "migrate")["command"] == [
        "/app/.venv/bin/python",
        "-m",
        "expense_tracker",
        "migrate",
    ]
    bot: Doc = container(spec, "bot")
    assert "command" not in bot
    assert "args" not in bot
    wait: Doc = container(spec, "wait-postgres")
    assert wait["image"] == postgres_image
    assert wait["command"] == [
        "sh",
        "-c",
        "until pg_isready -h postgres -p 5432 -U tracker -d tracker; do sleep 2; done",
    ]


def test_bot_images_and_pull_policy(docs: list[Doc]) -> None:
    spec: Doc = bot_spec(docs)
    names: list[str] = ["migrate", "bot"]
    name: str
    for name in names:
        c: Doc = container(spec, name)
        assert c["image"] == "expense-tracker:local"
        assert c["imagePullPolicy"] == "IfNotPresent"


def test_bot_and_migrate_environment(docs: list[Doc]) -> None:
    spec: Doc = bot_spec(docs)
    name: str
    for name in ("migrate", "bot"):
        c: Doc = container(spec, name)
        assert {"secretRef": {"name": SECRET}} in c["envFrom"]
        env: list[Doc] = c["env"]
        assert env[0]["name"] == "POSTGRES_PASSWORD"
        assert secret_key(env[0]) == {"name": SECRET, "key": "POSTGRES_PASSWORD"}
        assert env[1]["name"] == "DATABASE_URL"
        assert env[1]["value"] == "postgresql://tracker:$(POSTGRES_PASSWORD)@postgres:5432/tracker"


def test_bot_has_no_probes(docs: list[Doc]) -> None:
    bot: Doc = container(bot_spec(docs), "bot")
    assert "readinessProbe" not in bot
    assert "livenessProbe" not in bot


def test_postgres_image_is_the_compose_one_everywhere(docs: list[Doc], postgres_image: str) -> None:
    ci: Doc = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    assert ci["jobs"]["check"]["services"]["postgres"]["image"] == postgres_image
    assert container(pg_spec(docs), "postgres")["image"] == postgres_image
    assert container(bot_spec(docs), "wait-postgres")["image"] == postgres_image
    assert container(cron_spec(docs), "pg-backup")["image"] == postgres_image


def mount_volume(spec: Doc, c: Doc, path: str) -> tuple[Doc, Doc]:
    mounts: list[Doc] = [m for m in c["volumeMounts"] if m["mountPath"] == path]
    assert len(mounts) == 1, f"nothing mounted at {path}"
    volumes: list[Doc] = [v for v in spec.get("volumes", []) if v["name"] == mounts[0]["name"]]
    return mounts[0], volumes[0] if volumes else {}


def test_postgres_statefulset_storage(docs: list[Doc]) -> None:
    sts: Doc = one(docs, "StatefulSet", "postgres")
    spec: Doc = sts["spec"]
    assert spec["replicas"] == 1
    assert spec["serviceName"] == "postgres"
    claims: list[Doc] = [c for c in spec["volumeClaimTemplates"] if c["metadata"]["name"] == "data"]
    assert len(claims) == 1
    assert claims[0]["spec"]["accessModes"] == ["ReadWriteOnce"]
    assert claims[0]["spec"]["resources"]["requests"]["storage"] == "2Gi"
    pg: Doc = container(spec["template"]["spec"], "postgres")
    data_mounts: list[Doc] = [m for m in pg["volumeMounts"] if m["mountPath"] == "/var/lib/postgresql"]
    assert [m["name"] for m in data_mounts] == ["data"]
    mount: Doc
    volume: Doc
    mount, volume = mount_volume(spec["template"]["spec"], pg, "/backups")
    assert mount["readOnly"] is True
    assert volume["persistentVolumeClaim"]["claimName"] == "backups"


def test_postgres_probes(docs: list[Doc]) -> None:
    pg: Doc = container(pg_spec(docs), "postgres")
    kind: str
    for kind in ("readinessProbe", "livenessProbe"):
        command: list[str] = pg[kind]["exec"]["command"]
        joined: str = " ".join(command)
        assert "pg_isready" in joined
        assert re.search(r"-U\s+tracker", joined)
        assert re.search(r"-d\s+tracker", joined)
    assert pg["readinessProbe"]["periodSeconds"] == 5
    assert pg["livenessProbe"]["initialDelaySeconds"] == 30
    assert pg["livenessProbe"]["periodSeconds"] == 10
    assert pg["livenessProbe"]["failureThreshold"] == 6


def test_postgres_env(docs: list[Doc]) -> None:
    pg: Doc = container(pg_spec(docs), "postgres")
    env: dict[str, Doc] = {e["name"]: e for e in pg["env"]}
    assert env["POSTGRES_USER"]["value"] == "tracker"
    assert env["POSTGRES_DB"]["value"] == "tracker"
    assert secret_key(env["POSTGRES_PASSWORD"]) == {"name": SECRET, "key": "POSTGRES_PASSWORD"}


def test_postgres_service(docs: list[Doc]) -> None:
    svc: Doc = one(docs, "Service", "postgres")["spec"]
    assert svc["clusterIP"] == "None"
    assert [(p["port"], p["name"]) for p in svc["ports"]] == [(5432, "postgres")]
    labels: dict[str, str] = one(docs, "StatefulSet", "postgres")["spec"]["template"]["metadata"]["labels"]
    assert svc["selector"]
    assert svc["selector"].items() <= labels.items()


def test_pod_labels(docs: list[Doc]) -> None:
    expected: dict[str, str] = {"Deployment": "bot", "StatefulSet": "postgres", "CronJob": "pg-backup"}
    d: Doc
    for d in docs:
        if d["kind"] not in expected:
            continue
        template: Doc = (
            d["spec"]["jobTemplate"]["spec"]["template"] if d["kind"] == "CronJob" else d["spec"]["template"]
        )
        labels: dict[str, str] = template["metadata"]["labels"]
        assert labels["app.kubernetes.io/name"] == expected[d["kind"]]
        assert labels["app.kubernetes.io/part-of"] == NS


def test_every_container_has_resources(docs: list[Doc]) -> None:
    containers: list[Doc] = all_containers(docs)
    assert len(containers) == 5
    c: Doc
    for c in containers:
        res: Doc = c.get("resources", {})
        assert res.get("requests", {}).get("cpu"), c["name"]
        assert res.get("requests", {}).get("memory"), c["name"]
        assert res.get("limits", {}).get("cpu"), c["name"]
        assert res.get("limits", {}).get("memory"), c["name"]


def test_service_links_are_off(docs: list[Doc]) -> None:
    specs: list[tuple[str, Doc]] = pod_specs(docs)
    assert len(specs) == 3
    assert all(spec.get("enableServiceLinks") is False for _, spec in specs)


def test_no_literal_secrets_in_env(docs: list[Doc]) -> None:
    sensitive: re.Pattern[str] = re.compile("TOKEN|KEY|PASSWORD|SECRET", re.IGNORECASE)
    c: Doc
    for c in all_containers(docs):
        e: Doc
        for e in c.get("env", []):
            if sensitive.search(e["name"]):
                assert "value" not in e, f"{c['name']}: {e['name']} is a literal"
                assert "valueFrom" in e
            if e["name"] == "DATABASE_URL":
                match: re.Match[str] | None = re.search(r"://[^:/@]+:([^@]*)@", e["value"])
                assert match is not None
                assert match.group(1) == "$(POSTGRES_PASSWORD)"


def test_all_secret_references_point_at_the_one_secret(docs: list[Doc]) -> None:
    names: list[str] = []
    c: Doc
    for c in all_containers(docs):
        ref: Doc
        for ref in c.get("envFrom", []):
            if "secretRef" in ref:
                names.append(ref["secretRef"]["name"])
        e: Doc
        for e in c.get("env", []):
            if "secretKeyRef" in e.get("valueFrom", {}):
                names.append(e["valueFrom"]["secretKeyRef"]["name"])
    assert names
    assert set(names) == {SECRET}
    text: str = yaml.safe_dump(docs)
    assert "secretName" not in text


def test_images_are_pinned(docs: list[Doc]) -> None:
    containers: list[Doc] = all_containers(docs)
    assert containers
    c: Doc
    for c in containers:
        image: str = c["image"]
        last: str = image.rsplit("/", 1)[-1]
        assert ":" in last, f"{image} has no tag"
        assert not image.endswith(":latest")


def test_cronjob_pg_backup(docs: list[Doc]) -> None:
    cron: Doc = one(docs, "CronJob", "pg-backup")
    spec: Doc = cron["spec"]
    assert spec["schedule"] == "0 22 * * *"
    assert spec["timeZone"] == "Asia/Ho_Chi_Minh"
    assert spec["concurrencyPolicy"] == "Forbid"
    assert spec["startingDeadlineSeconds"] == 86400
    assert spec["successfulJobsHistoryLimit"] == 3
    assert spec["failedJobsHistoryLimit"] == 3
    assert spec["jobTemplate"]["spec"]["backoffLimit"] == 2
    assert spec["jobTemplate"]["spec"]["ttlSecondsAfterFinished"] == 86400
    pod: Doc = cron_spec(docs)
    assert pod["restartPolicy"] == "Never"
    c: Doc = container(pod, "pg-backup")
    assert c["command"] == ["bash", "/scripts/pg-backup.sh"]
    env: dict[str, Doc] = {e["name"]: e for e in c["env"]}
    assert env["BACKUP_KEEP"]["value"] == "14"
    assert env["BACKUP_DIR"]["value"] == "/backups"
    assert env["PGHOST"]["value"] == "postgres"
    assert env["PGPORT"]["value"] == "5432"
    assert env["PGUSER"]["value"] == "tracker"
    assert env["PGDATABASE"]["value"] == "tracker"
    assert secret_key(env["PGPASSWORD"]) == {"name": SECRET, "key": "POSTGRES_PASSWORD"}
    mount: Doc
    volume: Doc
    mount, volume = mount_volume(pod, c, "/backups")
    assert mount.get("readOnly") is not True
    assert volume["persistentVolumeClaim"]["claimName"] == "backups"
    mount, volume = mount_volume(pod, c, "/scripts")
    assert volume["configMap"]["name"] == configmap(docs)["metadata"]["name"]


def test_backups_pvc(docs: list[Doc]) -> None:
    spec: Doc = one(docs, "PersistentVolumeClaim", "backups")["spec"]
    assert spec["accessModes"] == ["ReadWriteOnce"]
    assert spec["resources"]["requests"]["storage"] == "2Gi"
    assert "storageClassName" not in spec


def test_configmap_carries_the_script_byte_for_byte(docs: list[Doc]) -> None:
    script: str = (ROOT / "k8s" / "base" / "pg-backup.sh").read_text()
    assert configmap(docs)["data"]["pg-backup.sh"] == script


def test_ci_has_independent_k8s_job() -> None:
    ci: Doc = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    job: Doc = ci["jobs"]["k8s"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert "needs" not in job
    text: str = yaml.safe_dump(job)
    assert "scripts/k3d-cluster.sh" in text
    assert "scripts/deploy.sh" in text
    assert "scripts/backup.sh" in text
    assert "scripts/restore.sh" in text
