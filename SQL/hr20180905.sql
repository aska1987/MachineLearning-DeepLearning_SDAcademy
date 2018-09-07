#��� ����� �ִ�޿� �����޿� �հ� ��ձ޿��� ���
select max(sal) maximum,min(sal) Minimum,sum(sal) Sum,round(avg(sal),2) Average
from emp;

#��� ���̺��� ��ձ޿��� ���� ���� �μ���ȣ�� ���
select deptno,round(avg(sal),2)
from emp
group by deptno;

#manager_id�� 120���� ���� ��� ����� ���� �Ʒ� ���Ǹ� �ۼ�
manager_id�� ��� �޿� �հ�� �� �޿��� ���� ���
select * from employees;
select manager_id, sum(salary)
from employees
where manager_id<120
group by rollup(manager_id);

#������̺��� ��� �޿��� 1800�̻��� �μ��� �μ���ȣ�� ��ձ޿��� ���
select deptno,round(avg(sal),2)
from emp
group by deptno
having avg(sal)>=1800;

#������̺� 1000�̻��� �޿��� �޴� ������� ��� �޿��� 1800�̻��� �μ��� �μ���ȣ�� ��ձ޿��� ���
select deptno,round(avg(sal),2)
from emp
where sal>=1000
group by deptno
having avg(sal)>=1800;

#�赵�� �������� �̸�,�޿�,�Ҽ��а�,�а���ġ�� ���
select * from professor;
select * from department;

select p.name,p.sal,d.deptno,d.loc
from professor p,department d
where p.deptno=d.deptno
and p.name='�赵��';

#1 �й�,�̸�,�а���ȣ,���������̸� ���
select * from student;
select * from professor;

select s.studno,s.name,s.deptno,p.name
from student s,professor p
where s.profno=p.profno;

#2 �й�,�̸�,�а���ȣ,���������̸�,�а���ġ ���
select s.studno,s.name,p.deptno,p.name,d.loc
from student s,professor p,department d
where s.profno=p.profno
and p.deptno=d.deptno;

#3 2����� ������ ������ �л��� ������
select s.studno,s.name,p.deptno,p.name,d.loc
from student s,professor p,department d
where s.profno=p.profno
and p.deptno=d.deptno
and s.name='������';

# �л� ���̺�� ���� ���̺��� �����Ͽ� �л��̸�,�г�,���������� �̸�, ������ ����ϼ���
select s.name,s.grade,p.name,p.position
from professor p join student s
using(profno);

#1. �� �޿��� $5000�� �Ѵ� �� ������ ���� ������ ���� �Ѿ��� president �� ���ܽ�Ű��, ���� �Ѿ׺��� ����Ʈ�� �����ϼ���
select * from emp;
select job,sum(nvl2(comm,sal+comm,sal)) payroll
from emp
where job <> 'PRESIDENT'
group by (job)
having sum(nvl2(comm,sal+comm,sal))>5000;
#2. rollup�����ڸ� �̿��Ͽ� �Ʒ��� ���� �μ���,������ ��ü ����� �� ��ü �޿��� �հ踦 ���
select * from emp;
select * from dept;

select p.dname,e.job,count(e.job) "total emp", sum(nvl2(e.comm,e.sal+e.comm,e.sal)) "total sal"
from emp e natural join dept p
group by rollup(dname,job);

#3. accounting �μ��� �ٹ��ϴ� ����� ���� last_name,����id,email,�μ���ȣ,�μ��̸��� last_name ������ ���
select * from employees;
select * from departments;
select last_name,job_id,email,department_id,department_name
from employees natural join departments
where department_name='Accounting'
order by last_name desc;
#4. 3���� �̾ accounting�μ��� �ٹ��ϴ¸������ ���� �̸�,����id,�μ��̸�,����,�ָ� ���
table:employees,departments,locations
select * from locations;
select last_name,job_id,department_name,city,state_province
from employees natural join departments
departments natural join locations
where department_name='Accounting';

#SMITH�� ���� �μ��� �ٹ��ϴ¸鼭 ADAMS���� �޿��� ���� ������ �̸�,�μ���ȣ,�޿��� ���
select * from emp;
select ename,deptno,sal
from emp
where deptno=(select deptno
             from emp
             where ename='SMITH')
and sal>(select sal
         from emp
         where ename='ADAMS');
         