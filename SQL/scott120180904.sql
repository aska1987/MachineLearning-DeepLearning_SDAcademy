#1. ��� ���̺��� �̸��� A�� �����ϰ� Ŀ�̼��� �޴� ����� �����ȣ,�̸�,�޿�,Ŀ�̼� ���
desc emp;
select * from emp;

select empno,ename,sal,comm
from emp
where ename like '%A%'
and comm is not null;

#1981�⿡ �Ի��� ��� ����� �̸��� �Ի����� ���ϴ� ���ǹ���?, ���������� ����
select ename,hiredate
from emp
where hiredate like '81%'
order by ename;

#���ʽ��� �޿��� 20%�̻��̰� �μ���ȣ�� 30�� ���� ��� ����� ���ؼ� �̸�,�޿� �׸��� ���ʽ��� ����ϴ� ���ǹ� �����϶�.
select ename,sal,comm
from emp
where comm>=sal*0.2
and deptno=30;
