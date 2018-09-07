#모든 사원의 최대급여 최저급여 합계 평균급여를 출력
select max(sal) maximum,min(sal) Minimum,sum(sal) Sum,round(avg(sal),2) Average
from emp;

#사원 테이블에서 평균급여가 가장 높은 부서번호를 출력
select deptno,round(avg(sal),2)
from emp
group by deptno;

#manager_id가 120보다 작은 모든 사원에 대해 아래 질의를 작성
manager_id별 사원 급여 합계와 총 급여의 합을 출력
select * from employees;
select manager_id, sum(salary)
from employees
where manager_id<120
group by rollup(manager_id);

#사원테이블에서 평균 급여가 1800이상인 부서의 부서번호와 평균급여를 출력
select deptno,round(avg(sal),2)
from emp
group by deptno
having avg(sal)>=1800;

#사원테이블에 1000이상의 급여를 받는 사원들의 평균 급여가 1800이상인 부서의 부서번호와 평균급여를 출력
select deptno,round(avg(sal),2)
from emp
where sal>=1000
group by deptno
having avg(sal)>=1800;

#김도훈 교수님의 이름,급여,소속학과,학과위치를 출력
select * from professor;
select * from department;

select p.name,p.sal,d.deptno,d.loc
from professor p,department d
where p.deptno=d.deptno
and p.name='김도훈';

#1 학번,이름,학과번호,지도교수이름 출력
select * from student;
select * from professor;

select s.studno,s.name,s.deptno,p.name
from student s,professor p
where s.profno=p.profno;

#2 학번,이름,학과번호,지도교수이름,학과위치 출력
select s.studno,s.name,p.deptno,p.name,d.loc
from student s,professor p,department d
where s.profno=p.profno
and p.deptno=d.deptno;

#3 2번출력 내용중 전인하 학생의 정보만
select s.studno,s.name,p.deptno,p.name,d.loc
from student s,professor p,department d
where s.profno=p.profno
and p.deptno=d.deptno
and s.name='전인하';

# 학생 테이블과 교수 테이블을 조인하여 학생이름,학년,지도교수의 이름, 직급을 출력하세요
select s.name,s.grade,p.name,p.position
from professor p join student s
using(profno);

#1. 총 급여가 $5000이 넘는 각 직무에 대해 직무와 월급 총액을 president 를 제외시키고, 월급 총액별로 리스트를 정렬하세요
select * from emp;
select job,sum(nvl2(comm,sal+comm,sal)) payroll
from emp
where job <> 'PRESIDENT'
group by (job)
having sum(nvl2(comm,sal+comm,sal))>5000;
#2. rollup연산자를 이용하여 아래와 같이 부서별,직업별 전체 사원수 및 전체 급여의 합계를 출력
select * from emp;
select * from dept;

select p.dname,e.job,count(e.job) "total emp", sum(nvl2(e.comm,e.sal+e.comm,e.sal)) "total sal"
from emp e natural join dept p
group by rollup(dname,job);

#3. accounting 부서에 근무하는 사원에 대해 last_name,직무id,email,부서번호,부서이름을 last_name 순으로 출력
select * from employees;
select * from departments;
select last_name,job_id,email,department_id,department_name
from employees natural join departments
where department_name='Accounting'
order by last_name desc;
#4. 3번에 이어서 accounting부서에 근무하는모든사원에 대해 이름,직무id,부서이름,도시,주를 출력
table:employees,departments,locations
select * from locations;
select last_name,job_id,department_name,city,state_province
from employees natural join departments
departments natural join locations
where department_name='Accounting';

#SMITH와 같은 부서에 근무하는면서 ADAMS보다 급여가 많은 직원의 이름,부서번호,급여를 출력
select * from emp;
select ename,deptno,sal
from emp
where deptno=(select deptno
             from emp
             where ename='SMITH')
and sal>(select sal
         from emp
         where ename='ADAMS');
         